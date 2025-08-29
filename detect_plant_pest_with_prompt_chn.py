"""
水稻病虫害识别程序，支持输出YOLO格式的病变位置坐标
专注于提高稻瘟病和细菌性枯萎病的识别准确率
"""

import os
import re
import base64
import json
import pandas as pd
import random
import glob
import logging
import asyncio
import aiohttp
import requests
from sklearn.model_selection import train_test_split
from PIL import Image, ImageEnhance
import pickle
from typing import Dict, List, Tuple, Optional
from string import Template


class RiceDiseaseAnalyzer:
    """
    水稻病虫害分析器，专注于提高特定病虫害的识别准确率
    支持输出YOLO格式的病变位置坐标
    """
    
    def __init__(self, config: Dict):
        # 加载配置
        self.config = config
        self.api_key = config['api']['api_key']
        self.model = config['api']['model']
        self.api_url = config['api']['api_url']
        self.max_batch_size = config['api']['max_batch_size']
        
        # 路径配置（处理变量替换）
        paths = config['paths']
        self.result_dir = paths['result_dir']
        self.cache_dir = self._resolve_path(paths['cache_dir'], paths)
        self.log_dir = self._resolve_path(paths['log_dir'], paths)
        self.data_dir = paths['data_dir']
        
        # 图像配置
        image_config = config['image']
        self.resized_image_size = tuple(image_config['resized_image_size'])
        self.contrast_enhance_ratio = image_config['contrast_enhance_ratio']
        self.image_quality = image_config['image_quality']
        
        # 训练配置
        training_config = config['training']
        self.target_diseases = training_config['target_diseases']
        self.retrain_threshold = training_config['retrain_threshold']
        self.sample_size = training_config['sample_size']
        self.test_size = training_config['test_size']
        self.random_state = training_config['random_state']
        self.force_retrain = training_config['force_retrain']
        
        # 预测配置
        prediction_config = config['prediction']
        self.low_confidence_threshold = prediction_config['low_confidence_threshold']
        self.max_tokens = prediction_config['max_tokens']
        self.prediction_max_tokens = prediction_config['prediction_max_tokens']
        
        # 初始化变量
        self.disease_types = self.target_diseases
        self.disease_index = {disease: idx for idx, disease in enumerate(self.target_diseases)}
        self.training_context = None
        self.key_features = {}
        
        
        # 确保目录存在
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
 
        # 配置日志
        self._setup_logging()
        
        # 缓存文件路径
        self.image_cache_file = os.path.join(self.cache_dir, "image_cache.pkl")
        self.context_cache_file = os.path.join(self.cache_dir, "training_context.pkl")
        self.image_cache = self._load_image_cache()
    
    def _resolve_path(self, path: str, paths: Dict) -> str:
        """解析包含变量的路径（如${result_dir}）"""
        return Template(path).substitute(paths)
    
    def _setup_logging(self) -> None:
        """配置日志系统"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.log_dir, 'plant_pest_analysis.log')),
                logging.StreamHandler()
            ]
        )
    
    def _preprocess_image(self, image_path: str) -> str:
        """预处理图像：增强对比度并保留细节"""
        try:
            with Image.open(image_path) as img:
                # 保持比例缩放
                img.thumbnail(self.resized_image_size)
                # 转换为RGB模式（去除alpha通道）
                if img.mode in ('RGBA', 'LA'):
                    background = Image.new(img.mode[:-1], img.size, (255, 255, 255))
                    background.paste(img, img.split()[-1])
                    img = background
                elif img.mode == 'P':
                    img = img.convert('RGB')
                
                # 增强对比度突出病斑特征
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(self.contrast_enhance_ratio)
                
                # 保存到内存并编码
                from io import BytesIO
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=self.image_quality)
                return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            logging.error(f"图像预处理失败 {image_path}: {str(e)}")
            # 预处理失败时使用原始编码
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _load_image_cache(self) -> Dict[str, str]:
        """加载图像Base64编码缓存"""
        if os.path.exists(self.image_cache_file):
            try:
                with open(self.image_cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logging.warning(f"加载图像缓存失败: {str(e)}，将创建新缓存")
        return {}
    
    def _save_image_cache(self) -> None:
        """保存图像Base64编码缓存"""
        try:
            with open(self.image_cache_file, 'wb') as f:
                pickle.dump(self.image_cache, f)
        except Exception as e:
            logging.error(f"保存图像缓存失败: {str(e)}")
    
    def encode_image(self, image_path: str) -> str:
        """将图像文件编码为Base64字符串用于API传输（带缓存）"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        # 检查缓存，使用绝对路径作为键
        abs_path = os.path.abspath(image_path)
        if abs_path in self.image_cache:
            return self.image_cache[abs_path]
        
        # 新图像编码并加入缓存
        base64_str = self._preprocess_image(image_path)
        self.image_cache[abs_path] = base64_str
        
        # 定期保存缓存
        if len(self.image_cache) % 10 == 0:
            self._save_image_cache()
        
        return base64_str
    
    def load_disease_images(self, root_dir: Optional[str] = None) -> Dict[str, List[str]]:
        """只加载目标类别的图像"""
        root_dir = root_dir or self.data_dir
        disease_data = {}
        
        # 仅加载目标病虫害类型
        for dir_name in self.target_diseases:
            dir_path = os.path.join(root_dir, dir_name)
            if os.path.isdir(dir_path):
                # 收集该病虫害下的所有图像
                image_files = []
                for ext in ['jpg', 'jpeg', 'png']:
                    image_files.extend(
                        glob.glob(os.path.join(dir_path, f'*.{ext}'), recursive=False)
                    )
                disease_data[dir_name] = image_files
                logging.info(f"加载目标类别 {dir_name} 图像 {len(image_files)} 张")
        
        # 检查是否有缺失的目标类别
        for disease in self.target_diseases:
            if disease not in disease_data:
                logging.warning(f"未找到 {disease} 的图像数据，请检查目录结构")
        
        return disease_data
    
    def split_train_val(self, disease_data: Dict[str, List[str]]) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """将目标类别的图像按比例分为训练集和验证集"""
        train_data = {}
        val_data = {}
        
        for disease, images in disease_data.items():
            if not images:
                continue
            # 划分训练集和验证集，随机种子确保结果可复现
            train_imgs, val_imgs = train_test_split(
                images, 
                test_size=self.test_size, 
                random_state=self.random_state
            )
            train_data[disease] = train_imgs
            val_data[disease] = val_imgs
            logging.info(f"{disease} 划分完成 - 训练集: {len(train_imgs)} 张, 验证集: {len(val_imgs)} 张")
        
        return train_data, val_data
    
    def save_training_context(self) -> None:
        """保存训练上下文到文件（包含关键特征）"""
        if self.training_context:
            try:
                save_data = {
                    "training_context": self.training_context,
                    "key_features": self.key_features,
                    "disease_index": self.disease_index
                }
                with open(self.context_cache_file, 'wb') as f:
                    pickle.dump(save_data, f)
                logging.info("训练上下文及关键特征已保存到缓存")
            except Exception as e:
                logging.error(f"保存训练上下文失败: {str(e)}")
    
    def load_training_context(self) -> bool:
        """从文件加载训练上下文及关键特征"""
        if os.path.exists(self.context_cache_file):
            try:
                with open(self.context_cache_file, 'rb') as f:
                    load_data = pickle.load(f)
                    self.training_context = load_data["training_context"]
                    self.key_features = load_data["key_features"]
                    self.disease_index = load_data.get("disease_index", {disease: idx for idx, disease in enumerate(self.target_diseases)})
                logging.info("已从缓存加载训练上下文及关键特征")
                return True
            except Exception as e:
                logging.warning(f"加载训练上下文失败: {str(e)}")
        return False
    
    def train_model(self, train_data: Dict[str, List[str]], force_retrain: Optional[bool] = None) -> Optional[str]:
        """
        专注于训练目标病虫害的区分特征
        增加样本量并优化提示词引导模型关注核心差异和病变位置
        """
        force_retrain = self.force_retrain if force_retrain is None else force_retrain
        # 如果不强制重训且缓存存在，则直接使用缓存
        if not force_retrain and self.load_training_context():
            return "已使用缓存的训练上下文"
        
        messages = []
        
        # 1. 任务说明与目标约束（包含位置标注要求）
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": 
                f"任务：精准学习并区分水稻的病虫害：{', '.join(self.target_diseases)}，并能标注病变位置。\n"
                f"类别索引：{json.dumps(self.disease_index, ensure_ascii=False)}\n"
                f"核心要求：\n"
                f"1. 发现各类病害的本质视觉差异，形成可直接用于判断的区分性特征\n"
                f"2. 学习准确定位病变区域，使用YOLO格式标注（类别索引 中心点x 中心点y 宽度 高度，坐标为归一化值0-1）\n"
                f"注意：只需要关注这些类别，无需考虑其他病虫害。"
            }]
        })
        
        # 2. 按类别展示样本并引导维度描述（包含位置特征）
        for disease, images in train_data.items():
            if not images:
                continue
            sample_size = min(self.sample_size, len(images))
            sample_images = random.sample(images, sample_size)
            disease_idx = self.disease_index.get(disease, 0)

            # 针对目标类别定制的特征分析维度
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": 
                    f"以下{len(sample_images)}张图像均为[{disease}]（类别索引：{disease_idx}）的典型样本，请从以下维度总结共性特征：\n"
                    f"1. 病斑初始形态（如针尖状、水渍状、圆形斑点等）；\n"
                    f"2. 病斑颜色变化（如初期黄色→后期褐色/灰白色，是否有深色边缘）；\n"
                    f"3. 病斑扩展模式（如沿叶脉蔓延、随机扩散、形成连续条带等）；\n"
                    f"4. 叶片质感变化（如病斑处是否凹陷、卷曲、有黏液/霉层）；\n"
                    f"5. 病斑位置特征（如主要分布在叶片尖端/中部/边缘，单个/多个聚集）；\n"
                    f"6. 典型病斑的YOLO格式坐标（类别索引 中心点x 中心点y 宽度 高度，坐标为归一化值）；\n"
                    f"7. 与其他类病的核心差异（包括位置和形态）。"
                }]
            })
            
            # 添加样本图像
            for img_path in sample_images:
                base64_img = self.encode_image(img_path)
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
                        }
                    ]
                })
        
        # 3. 要求结构化总结区分性特征（包含YOLO坐标示例）
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": 
                f"请对比{', '.join(self.target_diseases)}的特征，为每个类别总结5个最具区分性的标志，包括位置特征：\n"
                f"每个类别需补充典型病斑的YOLO格式坐标示例（类别索引按上述定义，坐标为归一化值）。\n"
                f"输出格式要求（必须严格遵循JSON格式）：\n"
                f"{{\n"
                f"  \"disease_features\": [\n"
                f"    {{\n"
                f"      \"disease_type\": \"病虫害名称\",\n"
                f"      \"index\": 类别索引,\n"
                f"      \"key_features\": [\n"
                f"        \"特征1（具体描述+与其他类的差异）\",\n"
                f"        \"特征2（位置描述：如主要分布在叶片尖端，YOLO示例：索引 0.3 0.2 0.1 0.1）\"\n"
                f"      ]\n"
                f"    }}\n"
                f"  ]\n"
                f"}}"
            }]
        })
        
        # 发送训练请求
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                data=json.dumps(payload)
            )
            response.raise_for_status()
            response_data = response.json()
            summary = response_data['choices'][0]['message']['content']
            if 'json' in summary:
                summary = summary.strip().replace('```json\n', '').replace('```', '')
            logging.info(f"模型特征总结:\n{summary}")
            
            # 解析并保存关键特征
            try:
                summary_json = json.loads(summary)
                for item in summary_json.get("disease_features", []):
                    self.key_features[item["disease_type"]] = {
                        "index": item.get("index", self.disease_index.get(item["disease_type"], 0)),
                        "features": item["key_features"]
                    }
            except json.JSONDecodeError:
                logging.warning(f"无法解析特征总结为JSON，将跳过关键特征提取.{summary}")
            
            # 保存训练后的上下文
            self.training_context = messages + [
                {"role": "assistant", "content": summary}
            ]
            self.save_training_context()  # 保存到缓存
            return summary
            
        except Exception as e:
            logging.error(f"训练模型时发生错误: {str(e)}")
            return None
    
    async def _async_predict(self, session: aiohttp.ClientSession, image_path: str) -> Dict[str, str]:
        """异步预测单张图像，强制关联训练得到的关键特征，输出YOLO格式坐标"""
        if not self.training_context:
            return {
                "disease_type": "未知",
                "confidence": "0%",
                "reason": "模型尚未训练",
                "yolo_annotations": []
            }
        
        # 构建特征提示文本
        feature_prompt = "类别索引及关键特征：\n"
        for disease, features_info in self.key_features.items():
            feature_prompt += f"{disease}（索引{features_info['index']}）：{'; '.join(features_info['features'])}\n"
        
        # 复制训练上下文
        messages = self.training_context.copy()
        
        # 添加待预测图像及提示（关联关键特征，要求输出YOLO坐标）
        base64_img = self.encode_image(image_path)
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"请根据以下关键特征判断图像属于哪种病虫害（仅{', '.join(self.target_diseases)}中选择），并输出病变位置：\n"
                            f"{feature_prompt}\n"
                            f"判断要求：\n"
                            f"1. 若存在病变，必须明确对应上述哪条特征，并说明图像中哪些细节符合该特征；\n"
                            f"2. 必须输出病变区域的YOLO格式坐标（一行一个病变，无病变则输出空列表）：\n"
                            f"   - 格式：类别索引 中心点x 中心点y 宽度 高度（坐标为相对于图像宽高的归一化值，范围0-1）；\n"
                            f"   - 若有多个病变，每行一个坐标；\n"
                            f"3. 若特征匹配度低（<{self.low_confidence_threshold}%），请明确说明不确定的原因。\n"
                            f"仅输出JSON字符串，不添加任何额外解释、说明或格式标记（如```json）\n"
                            f"{{\n"
                            f"  'disease_type': '病虫害名称（或无病变）',\n"
                            f"  'confidence': '可信度(0-100%)',\n"
                            f"  'reason': '具体特征匹配说明',\n"
                            f"  'yolo_annotations': ['类别索引 x_center y_center width height', ...]  # 空列表表示无病变\n"
                            f"}}"
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
                }
            ]
        })
        
        # 发送预测请求
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.prediction_max_tokens
        }
        
        try:
            async with session.post(
                self.api_url,
                headers=headers,
                data=json.dumps(payload)
            ) as response:
                response.raise_for_status()
                response_data = await response.json()
                content = response_data['choices'][0]['message']['content']
                result = self._parse_prediction(content)
                
                # 低置信度二次判断
                confidence = self._parse_confidence(result["confidence"])
                if confidence < self.low_confidence_threshold:
                    return await self._low_confidence_recheck(session, image_path, result)
                return result
                
        except Exception as e:
            logging.error(f"预测图像 {os.path.basename(image_path)} 时发生错误: {str(e)}")
            return {
                "disease_type": "未知",
                "confidence": "0%",
                "reason": f"预测错误: {str(e)}",
                "yolo_annotations": []
            }
    
    async def _low_confidence_recheck(self, session: aiohttp.ClientSession, image_path: str, initial_result: Dict) -> Dict:
        """对低置信度结果进行二次检查，重点校准YOLO坐标"""
        messages = self.training_context.copy()
        base64_img = self.encode_image(image_path)
        
        # 构建二次检查提示，强调YOLO坐标准确性
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"之前预测为{initial_result['disease_type']}（可信度{initial_result['confidence']}），但可信度较低。\n"
                            f"之前标注的病变位置：{initial_result['yolo_annotations']}\n"
                            f"请重新检查以下关键差异点及病变位置：\n"
                            f"1. 病斑中心是否有灰白色区域（稻瘟病典型特征）及位置；\n"
                            f"2. 是否有沿叶脉蔓延的水渍状条带（细菌性枯萎病典型特征）及蔓延路径；\n"
                            f"3. 病斑边缘是否有明显褐色晕圈及边界范围；\n"
                            f"4. 重新校准YOLO坐标：确保中心点和宽高准确反映病斑实际位置（归一化值）；\n"
                            f"请基于以上细节重新判断并提高描述精度，YOLO坐标需严格校验。\n"
                            f"仅输出JSON字符串，不添加任何额外解释、说明或格式标记（如```json）\n"
                            f"返回格式同上，必须包含'yolo_annotations'字段。"
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
                }
            ]
        })
        
        try:
            async with session.post(
                self.api_url,
                headers={"Content-Type": "application/json", 
                         "Authorization": f"Bearer {self.api_key}" if self.api_key else ""},
                data=json.dumps({"model": self.model, "messages": messages, "max_tokens": self.prediction_max_tokens})
            ) as response:
                response.raise_for_status()
                response_data = await response.json()
                content = response_data['choices'][0]['message']['content']
                return self._parse_prediction(content)
        except Exception as e:
            logging.warning(f"二次检查失败: {str(e)}，将返回初始结果")
            return initial_result
    
    def _parse_confidence(self, confidence_str: str) -> int:
        """解析置信度字符串为整数"""
        try:
            return int(re.search(r'\d+', confidence_str).group())
        except:
            return 0
    
    async def _batch_predict_async(self, image_paths: List[str]) -> List[Dict[str, str]]:
        """异步批量预测图像"""
        async with aiohttp.ClientSession() as session:
            tasks = [self._async_predict(session, path) for path in image_paths]
            return await asyncio.gather(*tasks)
    
    def batch_predict(self, image_paths: List[str]) -> List[Dict[str, str]]:
        """批量预测图像的病虫害类型，返回包含YOLO坐标的结果"""
        if not image_paths:
            return []
            
        # 分块处理，避免单次请求过多
        results = []
        for i in range(0, len(image_paths), self.max_batch_size):
            batch = image_paths[i:i+self.max_batch_size]
            logging.info(f"处理批量预测: {i+1}-{min(i+self.max_batch_size, len(image_paths))}/{len(image_paths)}")
            batch_results = asyncio.run(self._batch_predict_async(batch))
            results.extend(batch_results)
        
        return results
    
    def predict_disease(self, image_path: str) -> Dict[str, str]:
        """使用训练好的模型预测单张图像的病虫害类型，返回包含YOLO坐标的结果"""
        return self.batch_predict([image_path])[0]
    
    def _parse_prediction(self, content: str) -> Dict[str, str]:
        """解析模型返回的预测结果，特别处理YOLO坐标"""
        try:
            # 尝试直接解析JSON
            result = json.loads(content)
            # 确保yolo_annotations字段存在且为列表
            yolo_annotations = result.get("yolo_annotations", [])
            if not isinstance(yolo_annotations, list):
                yolo_annotations = []
                
            return {
                "disease_type": result.get("disease_type", "未知"),
                "confidence": result.get("confidence", "0%"),
                "reason": result.get("reason", "无判断依据"),
                "yolo_annotations": yolo_annotations
            }
        except json.JSONDecodeError:
            # 如果不是标准JSON，尝试从文本中提取信息
            disease_type = "未知"
            for dt in self.target_diseases:
                if dt in content:
                    disease_type = dt
                    break
            
            # 尝试提取置信度
            confidence = "未知"
            match = re.search(r'(\d+)%', content)
            if match:
                confidence = f"{match.group(1)}%"
            
            # 尝试提取YOLO格式坐标
            yolo_pattern = r'\b\d+\s+0\.\d+\s+0\.\d+\s+0\.\d+\s+0\.\d+\b'
            yolo_matches = re.findall(yolo_pattern, content)
            
            return {
                "disease_type": disease_type,
                "confidence": confidence,
                "reason": content,
                "yolo_annotations": yolo_matches
            }
    
    def save_yolo_labels(self, image_paths: List[str], results: List[Dict], output_dir: str) -> None:
        """将预测结果保存为YOLO格式的标签文件"""
        os.makedirs(output_dir, exist_ok=True)
        
        for img_path, result in zip(image_paths, results):
            # 获取图像文件名（不含扩展名）作为标签文件名
            img_basename = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(output_dir, f"{img_basename}.txt")
            
            # 写入YOLO格式标签
            with open(label_path, 'w', encoding='utf-8') as f:
                for annotation in result.get("yolo_annotations", []):
                    # 确保每行只有一个标注
                    f.write(f"{annotation}\n")
        
        logging.info(f"已将{len(image_paths)}个预测结果保存为YOLO格式标签至 {output_dir}")
    
    def evaluate_accuracy(self, val_data: Dict[str, List[str]], result_dir: Optional[str] = None) -> Dict:
        """评估模型在验证集上的准确度并收集错误案例，包含YOLO坐标评估"""
        result_dir = result_dir or self.result_dir
        results = []
        total_correct = 0
        total_samples = 0
        error_samples = []  # 收集错误案例
        
        # 创建结果保存目录
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        
        # 创建YOLO标签保存目录
        yolo_label_dir = os.path.join(result_dir, "yolo_labels")
        os.makedirs(yolo_label_dir, exist_ok=True)
        
        # 收集所有验证图像及对应的真实标签
        all_images = []
        all_true_labels = []
        
        for true_disease, images in val_data.items():
            for img_path in images:
                all_images.append(img_path)
                all_true_labels.append(true_disease)
                total_samples += 1
        
        # 批量预测所有验证图像
        logging.info(f"开始批量预测 {total_samples} 张验证图像...")
        predictions = self.batch_predict(all_images)
        
        # 保存YOLO格式标签
        self.save_yolo_labels(all_images, predictions, yolo_label_dir)
        
        # 处理预测结果
        for img_path, true_disease, prediction in zip(all_images, all_true_labels, predictions):
            img_name = os.path.basename(img_path)
            predicted_disease = prediction["disease_type"]
            
            # 判断是否预测正确
            is_correct = (predicted_disease == true_disease)
            if is_correct:
                total_correct += 1
            else:
                # 收集错误案例
                error_samples.append({
                    "image_path": img_path,
                    "true_label": true_disease,
                    "wrong_prediction": predicted_disease,
                    "reason": prediction["reason"],
                    "yolo_annotations": prediction["yolo_annotations"]
                })
            
            # 保存结果
            results.append({
                "image_name": img_name,
                "true_disease": true_disease,
                "predicted_disease": predicted_disease,
                "confidence": prediction["confidence"],
                "reason": prediction["reason"],
                "yolo_annotations": prediction["yolo_annotations"],
                "is_correct": is_correct
            })
            
            if total_samples <= 100 or total_samples % 100 == 0:
                logging.info(
                    f"图像 {img_name} - 真实: {true_disease}, 预测: {predicted_disease}, "
                    f"{'正确' if is_correct else '错误'}"
                )
        
        # 计算总体准确度
        accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0
        logging.info(f"\n总体识别准确度: {accuracy:.2f}% ({total_correct}/{total_samples})")
        
        # 按类别计算准确度
        class_accuracy = {}
        for disease in self.target_diseases:
            class_samples = [r for r in results if r["true_disease"] == disease]
            if class_samples:
                correct = sum(1 for r in class_samples if r["is_correct"])
                class_accuracy[disease] = (correct / len(class_samples)) * 100
                logging.info(f"{disease} 识别准确度: {class_accuracy[disease]:.2f}% ({correct}/{len(class_samples)})")
        
        # 保存结果到Excel和JSON
        results_df = pd.DataFrame(results)
        import time
        timestamp = time.strftime("%Y%m%d%H%M%S")
        results_df.to_excel(os.path.join(result_dir, f"prediction_results_{timestamp}.xlsx"), index=False)
        with open(os.path.join(result_dir, f"prediction_results_{timestamp}.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        
        # 保存错误案例
        if error_samples:
            with open(os.path.join(result_dir, f"error_samples_{timestamp}.json"), "w", encoding="utf-8") as f:
                json.dump(error_samples, f, ensure_ascii=False, indent=4)
            logging.info(f"已保存 {len(error_samples)} 个错误案例用于二次训练")
        
        # 保存图像缓存
        self._save_image_cache()
        
        return {
            "overall_accuracy": accuracy,
            "class_accuracy": class_accuracy,
            "results": results_df,
            "error_samples": error_samples,
            "yolo_label_dir": yolo_label_dir
        }
    
    def retrain_with_errors(self, error_samples: List[Dict]) -> Optional[str]:
        """使用错误案例进行二次训练，提高识别精度和YOLO坐标准确性"""
        if not error_samples or not self.training_context:
            logging.warning("没有错误案例或模型未训练，无法进行二次训练")
            return None
        
        logging.info(f"使用 {len(error_samples)} 个错误案例进行二次训练...")
        messages = self.training_context.copy()
        
        # 添加错误案例分析提示，强调坐标准确性
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": 
                "以下是之前预测错误的案例，请分析误判原因并修正特征总结和坐标标注：\n"
                "要求：\n"
                "1. 明确指出每个案例中误判的关键特征混淆点\n"
                "2. 分析病变位置标注错误的原因（如边界不准确、中心点偏移等）\n"
                "3. 补充或修正之前总结的区分性特征和典型位置特征\n"
                "4. 输出格式保持与之前相同的JSON结构，包含正确的YOLO坐标示例"
            }]
        })
        
        # 添加错误案例图像及说明
        for i, error in enumerate(error_samples[:10]):  # 每次最多使用10个案例，避免输入过长
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": 
                    f"错误案例 {i+1}：实际为{error['true_label']}，被误判为{error['wrong_prediction']}\n"
                    f"原判断依据：{error['reason']}\n"
                    f"原病变位置标注：{error['yolo_annotations']}\n"
                    f"请重点分析：为什么会误判？病变位置标注有何问题？正确的标注应该是什么？"
                }]
            })
            
            base64_img = self.encode_image(error["image_path"])
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
                    }
                ]
            })
        
        # 发送二次训练请求
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                data=json.dumps(payload)
            )
            response.raise_for_status()
            response_data = response.json()
            summary = response_data['choices'][0]['message']['content']
            if 'json' in summary:
                summary = summary.strip().replace('```json\n', '').replace('```', '')
            logging.info(f"二次训练特征修正:\n{summary}")
            
            # 更新关键特征
            try:
                summary_json = json.loads(summary)
                for item in summary_json.get("disease_features", []):
                    self.key_features[item["disease_type"]] = {
                        "index": item.get("index", self.disease_index.get(item["disease_type"], 0)),
                        "features": item["key_features"]
                    }
            except json.JSONDecodeError:
                logging.warning(f"无法解析二次训练总结为JSON{summary}")
            
            # 更新训练上下文
            self.training_context = messages + [
                {"role": "assistant", "content": summary}
            ]
            self.save_training_context()
            return summary
            
        except Exception as e:
            logging.error(f"二次训练时发生错误: {str(e)}")
            return None


def load_config(config_path: str = "") -> Dict:
    """加载配置文件"""
    if config_path  == "":
        # 获取当前程序文件的目录
        program_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(program_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


if __name__ == "__main__":
    # 加载配置文件
    config = load_config()
    
    # 初始化分析器
    analyzer = RiceDiseaseAnalyzer(config)
    
    # 加载数据
    disease_data = analyzer.load_disease_images()
    
    # 划分训练集和验证集
    train_data, val_data = analyzer.split_train_val(disease_data)
    
    # 训练模型（首次训练或强制重训）
    analyzer.train_model(train_data)
    
    # 评估模型并获取错误案例
    eval_result = analyzer.evaluate_accuracy(val_data)
    logging.info(f"YOLO格式标签已保存至: {eval_result.get('yolo_label_dir', '')}")
    
    # 筛选需要二次训练的错误案例（仅针对精度低于阈值的类别）
    class_accuracy = eval_result["class_accuracy"]
    filtered_errors = [
        err for err in eval_result["error_samples"]
        if class_accuracy.get(err["true_label"], 0) < analyzer.retrain_threshold
    ]
    
    # 使用筛选后的错误案例进行二次训练
    if filtered_errors:
        # 打印需要优化的类别及其精度
        need_improve = set(err["true_label"] for err in filtered_errors)
        for disease in need_improve:
            logging.info(f"类别 {disease} 精度 {class_accuracy[disease]:.2f}% 低于阈值，将进行二次训练")
        
        analyzer.retrain_with_errors(filtered_errors)
        
        # 二次评估
        analyzer.evaluate_accuracy(val_data, os.path.join(analyzer.result_dir, "after_retrain"))
    else:
        logging.info(f"所有类别的精度均不低于{analyzer.retrain_threshold}%，无需进行二次训练")
