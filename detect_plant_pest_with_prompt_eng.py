"""
Rice disease and pest identification program, supporting output of lesion coordinates in YOLO format
Focus on improving recognition accuracy for specific rice diseases such as rice blast and bacterial blight
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


class ConfigLoader:
    """Configuration loader with support for path variable substitution"""
    @staticmethod
    def load(config_path: str = None) -> Dict:
        """Load configuration file from the same directory as the program if not specified"""
        # 如果未指定路径，自动使用程序所在目录下的config.json
        if config_path is None:
            # 获取当前程序文件的目录
            program_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(program_dir, "config.json")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 解析路径变量（保持原有逻辑）
        ConfigLoader._resolve_path_variables(config, config)
        return config
    @staticmethod
    def _resolve_path_variables(config: Dict, root_config: Dict, parent_key: str = "") -> None:
        """Recursively resolve path variables in configuration"""
        for key, value in config.items():
            current_key = f"{parent_key}.{key}" if parent_key else key
            
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                # Extract variable path (e.g., ${paths.result_dir} -> paths.result_dir)
                var_path = value[2:-1]
                var_parts = var_path.split('.')
                
                # Get variable value from root configuration
                var_value = root_config
                try:
                    for part in var_parts:
                        var_value = var_value[part]
                    config[key] = var_value
                except (KeyError, TypeError):
                    raise ValueError(f"Failed to resolve configuration variable: {value}")
            
            elif isinstance(value, dict):
                # Recursively process sub-dictionaries
                ConfigLoader._resolve_path_variables(value, root_config, current_key)


# Load configuration
CONFIG = ConfigLoader.load()

# Initialize directories
os.makedirs(CONFIG["paths"]["log_dir"], exist_ok=True)
os.makedirs(CONFIG["paths"]["cache_dir"], exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(CONFIG["paths"]["log_dir"], 'plant_pest_analysis.log')),
        logging.StreamHandler()
    ]
)


class RiceDiseaseAnalyzer:
    """
    Rice disease and pest analyzer, focusing on improving recognition accuracy for specific diseases
    Supports output of lesion coordinates in YOLO format
    """
    
    def __init__(self):
        # Load API parameters from configuration
        self.api_key = CONFIG["api"]["api_key"]
        self.model = CONFIG["api"]["model"]
        self.api_url = CONFIG["api"]["api_url"]
        self.max_batch_size = CONFIG["api"]["max_batch_size"]
        
        # Load training parameters from configuration
        self.disease_types = CONFIG["training"]["target_diseases"]
        self.disease_index = {disease: idx for idx, disease in enumerate(self.disease_types)}
        self.retrain_threshold = CONFIG["training"]["retrain_threshold"]
        self.force_retrain = CONFIG["training"]["force_retrain"]
        
        # Other configuration parameters
        self.resized_image_size = tuple(CONFIG["image"]["resized_image_size"])
        self.contrast_ratio = CONFIG["image"]["contrast_enhance_ratio"]
        self.image_quality = CONFIG["image"]["image_quality"]
        self.low_confidence_threshold = CONFIG["prediction"]["low_confidence_threshold"]
        
        # State variables
        self.training_context = None
        self.key_features = {}
        self.image_cache = self._load_image_cache()
        
        # Cache file paths
        self.image_cache_file = os.path.join(CONFIG["paths"]["cache_dir"], "image_cache.pkl")
        self.context_cache_file = os.path.join(CONFIG["paths"]["cache_dir"], "training_context.pkl")
    
    def _preprocess_image(self, image_path: str) -> str:
        """Preprocess image: enhance contrast while preserving details"""
        try:
            with Image.open(image_path) as img:
                # Resize while maintaining aspect ratio
                img.thumbnail(self.resized_image_size)
                
                # Convert to RGB mode (remove alpha channel)
                if img.mode in ('RGBA', 'LA'):
                    background = Image.new(img.mode[:-1], img.size, (255, 255, 255))
                    background.paste(img, img.split()[-1])
                    img = background
                elif img.mode == 'P':
                    img = img.convert('RGB')
                
                # Enhance contrast to highlight lesion features
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(self.contrast_ratio)
                
                # Save to memory and encode
                from io import BytesIO
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=self.image_quality)
                return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            logging.error(f"Image preprocessing failed {image_path}: {str(e)}")
            # Use original encoding if preprocessing fails
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _load_image_cache(self) -> Dict[str, str]:
        """Load image Base64 encoding cache"""
        cache_file = os.path.join(CONFIG["paths"]["cache_dir"], "image_cache.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logging.warning(f"Failed to load image cache: {str(e)}, creating new cache")
        return {}
    
    def _save_image_cache(self) -> None:
        """Save image Base64 encoding cache"""
        try:
            with open(self.image_cache_file, 'wb') as f:
                pickle.dump(self.image_cache, f)
        except Exception as e:
            logging.error(f"Failed to save image cache: {str(e)}")
    
    def encode_image(self, image_path: str) -> str:
        """Encode image file to Base64 string for API transmission (with caching)"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Check cache, use absolute path as key
        abs_path = os.path.abspath(image_path)
        if abs_path in self.image_cache:
            return self.image_cache[abs_path]
        
        # Encode new image and add to cache
        base64_str = self._preprocess_image(image_path)
        self.image_cache[abs_path] = base64_str
        
        # Save cache periodically
        if len(self.image_cache) % 10 == 0:
            self._save_image_cache()
        
        return base64_str
    
    def load_disease_images(self, root_dir: str = None) -> Dict[str, List[str]]:
        """Load only images of target categories, default to data directory in configuration"""
        root_dir = root_dir or CONFIG["paths"]["data_dir"]
        disease_data = {}
        
        # Load only target disease types
        for dir_name in self.disease_types:
            dir_path = os.path.join(root_dir, dir_name)
            if os.path.isdir(dir_path):
                # Collect all images under this disease category
                image_files = []
                for ext in ['jpg', 'jpeg', 'png']:
                    image_files.extend(
                        glob.glob(os.path.join(dir_path, f'*.{ext}'), recursive=False)
                    )
                disease_data[dir_name] = image_files
                logging.info(f"Loaded {len(image_files)} images for target category {dir_name}")
        
        # Check for missing target categories
        for disease in self.disease_types:
            if disease not in disease_data:
                logging.warning(f"No image data found for {disease}, please check directory structure")
        
        return disease_data
    
    def split_train_val(self, disease_data: Dict[str, List[str]]) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """Split images of target categories into training and validation sets by ratio"""
        train_data = {}
        val_data = {}
        test_size = CONFIG["training"]["test_size"]
        random_state = CONFIG["training"]["random_state"]
        
        for disease, images in disease_data.items():
            if not images:
                continue
            # Split into training and validation sets, random seed ensures reproducibility
            train_imgs, val_imgs = train_test_split(
                images, 
                test_size=test_size, 
                random_state=random_state
            )
            train_data[disease] = train_imgs
            val_data[disease] = val_imgs
            logging.info(f"{disease} split completed - Training set: {len(train_imgs)} images, Validation set: {len(val_imgs)} images")
        
        return train_data, val_data
    
    def save_training_context(self) -> None:
        """Save training context to file (including key features)"""
        if self.training_context:
            try:
                save_data = {
                    "training_context": self.training_context,
                    "key_features": self.key_features,
                    "disease_index": self.disease_index
                }
                with open(self.context_cache_file, 'wb') as f:
                    pickle.dump(save_data, f)
                logging.info("Training context and key features saved to cache")
            except Exception as e:
                logging.error(f"Failed to save training context: {str(e)}")
    
    def load_training_context(self) -> bool:
        """Load training context and key features from file"""
        if os.path.exists(self.context_cache_file):
            try:
                with open(self.context_cache_file, 'rb') as f:
                    load_data = pickle.load(f)
                    self.training_context = load_data["training_context"]
                    self.key_features = load_data["key_features"]
                    self.disease_index = load_data.get("disease_index", {disease: idx for idx, disease in enumerate(self.disease_types)})
                logging.info("Loaded training context and key features from cache")
                return True
            except Exception as e:
                logging.warning(f"Failed to load training context: {str(e)}")
        return False
    
    def train_model(self, train_data: Dict[str, List[str]]) -> Optional[str]:
        """
        Focus on training distinguishing features of target diseases and pests
        Increase sample size and optimize prompts to guide model focus on core differences and lesion locations
        """
        sample_size = CONFIG["training"]["sample_size"]
        # Use cache if not forcing retraining and cache exists
        if not self.force_retrain and self.load_training_context():
            return "Using cached training context"
        
        messages = []
        
        # 1. Task description and target constraints (including location annotation requirements)
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": 
                f"Task: Precisely learn and distinguish rice diseases and pests: {', '.join(self.disease_types)}, and be able to annotate lesion locations.\n"
                f"Category indices: {json.dumps(self.disease_index, ensure_ascii=False)}\n"
                f"Core requirements:\n"
                f"1. Discover essential visual differences between various diseases, forming distinguishable features directly usable for judgment\n"
                f"2. Learn to accurately locate lesion areas using YOLO format annotation (category index center_x center_y width height, coordinates are normalized values 0-1)\n"
                f"Note: Only focus on these categories, no need to consider other diseases and pests."
            }]
        })
        
        # 2. Display samples by category and guide dimensional description (including location features)
        for disease, images in train_data.items():
            if not images:
                continue
            sample_count = min(sample_size, len(images))
            sample_images = random.sample(images, sample_count)
            disease_idx = self.disease_index.get(disease, 0)

            # Custom feature analysis dimensions for target categories
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": 
                    f"The following {len(sample_images)} images are typical samples of [{disease}] (category index: {disease_idx}), please summarize common features from the following dimensions:\n"
                    f"1. Initial lesion morphology (e.g., pinpoint-like, water-soaked, circular spots, etc.);\n"
                    f"2. Lesion color changes (e.g., initial yellow → later brown/grayish white, presence of dark edges);\n"
                    f"3. Lesion expansion pattern (e.g., spreading along leaf veins, random diffusion, forming continuous strips, etc.);\n"
                    f"4. Leaf texture changes (e.g., whether lesion area is sunken, curled, has mucus/mold layer);\n"
                    f"5. Lesion location features (e.g., mainly distributed at leaf tip/middle/edge, single/multiple clusters);\n"
                    f"6. YOLO format coordinates of typical lesions (category index center_x center_y width height, coordinates are normalized values);\n"
                    f"7. Core differences from other disease categories (including location and morphology)."
                }]
            })
            
            # Add sample images
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
        
        # 3. Request structured summary of distinguishing features (including YOLO coordinate examples)
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": 
                f"Please compare features of {', '.join(self.disease_types)}, and summarize 5 most distinguishing markers for each category, including location features:\n"
                f"Each category should include YOLO format coordinate examples of typical lesions (category indices as defined above, coordinates are normalized values).\n"
                f"Output format requirements (must strictly follow JSON format):\n"
                f"{{\n"
                f"  \"disease_features\": [\n"
                f"    {{\n"
                f"      \"disease_type\": \"disease name\",\n"
                f"      \"index\": category index,\n"
                f"      \"key_features\": [\n"
                f"        \"Feature 1 (specific description + differences from other categories)\",\n"
                f"        \"Feature 2 (location description: e.g., mainly distributed at leaf tip, YOLO example: index 0.3 0.2 0.1 0.1)\"\n"
                f"      ]\n"
                f"    }}\n"
                f"  ]\n"
                f"}}"
            }]
        })
        
        # Send training request
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": CONFIG["prediction"]["max_tokens"]
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
            logging.info(f"Model feature summary:\n{summary}")
            
            # Parse and save key features
            try:
                summary_json = json.loads(summary)
                for item in summary_json.get("disease_features", []):
                    self.key_features[item["disease_type"]] = {
                        "index": item.get("index", self.disease_index.get(item["disease_type"], 0)),
                        "features": item["key_features"]
                    }
            except json.JSONDecodeError:
                logging.warning(f"Failed to parse feature summary as JSON, skipping key feature extraction.{summary}")
            
            # Save trained context
            self.training_context = messages + [
                {"role": "assistant", "content": summary}
            ]
            self.save_training_context()  # Save to cache
            return summary
            
        except Exception as e:
            logging.error(f"Error occurred while training model: {str(e)}")
            return None
    
    async def _async_predict(self, session: aiohttp.ClientSession, image_path: str) -> Dict[str, str]:
        """Asynchronously predict a single image, enforce association with trained key features, output YOLO format coordinates"""
        if not self.training_context:
            return {
                "disease_type": "Unknown",
                "confidence": "0%",
                "reason": "Model not trained yet",
                "yolo_annotations": []
            }
        
        # Build feature prompt text
        feature_prompt = "Category indices and key features:\n"
        for disease, features_info in self.key_features.items():
            feature_prompt += f"{disease} (index {features_info['index']}): {'; '.join(features_info['features'])}\n"
        
        # Copy training context
        messages = self.training_context.copy()
        
        # Add image to predict and prompt (associate with key features, require YOLO coordinates)
        base64_img = self.encode_image(image_path)
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Please determine which disease the image belongs to based on the following key features (choose only from {', '.join(self.disease_types)}), and output lesion locations:\n"
                            f"{feature_prompt}\n"
                            f"Judgment requirements:\n"
                            f"1. If lesions exist, must clearly correspond to which of the above features, and explain which details in the image match the feature;\n"
                            f"2. Must output YOLO format coordinates of lesion areas (one lesion per line, output empty list if no lesions):\n"
                            f"   - Format: category_index center_x center_y width height (coordinates are normalized values relative to image width and height, range 0-1);\n"
                            f"   - If multiple lesions, one coordinate per line;\n"
                            f"3. If feature matching degree is low (<{self.low_confidence_threshold}%), clearly explain the reasons for uncertainty.\n"
                            f"Output only JSON string without any additional explanations or format markers (such as ```json)\n"
                            f"{{\n"
                            f"  'disease_type': 'disease name (or no lesion)',\n"
                            f"  'confidence': 'confidence level (0-100%)',\n"
                            f"  'reason': 'specific feature matching explanation',\n"
                            f"  'yolo_annotations': ['category_index x_center y_center width height', ...]  # empty list indicates no lesion\n"
                            f"}}"
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
                }
            ]
        })
        
        # Send prediction request
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": CONFIG["prediction"]["prediction_max_tokens"]
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
                
                # Low confidence secondary judgment
                confidence = self._parse_confidence(result["confidence"])
                if confidence < self.low_confidence_threshold:
                    return await self._low_confidence_recheck(session, image_path, result)
                return result
                
        except Exception as e:
            logging.error(f"Error predicting image {os.path.basename(image_path)}: {str(e)}")
            return {
                "disease_type": "Unknown",
                "confidence": "0%",
                "reason": f"Prediction error: {str(e)}",
                "yolo_annotations": []
            }
    
    async def _low_confidence_recheck(self, session: aiohttp.ClientSession, image_path: str, initial_result: Dict) -> Dict:
        """Recheck low confidence results, focusing on calibrating YOLO coordinates"""
        messages = self.training_context.copy()
        base64_img = self.encode_image(image_path)
        
        # Build secondary check prompt, emphasizing YOLO coordinate accuracy
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Previous prediction was {initial_result['disease_type']} (confidence {initial_result['confidence']}), but confidence is low.\n"
                            f"Previously annotated lesion locations: {initial_result['yolo_annotations']}\n"
                            f"Please recheck the following key differences and lesion locations:\n"
                            f"1. Whether there is a grayish white area in the lesion center (typical feature of rice blast) and its location;\n"
                            f"2. Whether there are yellow strips distributed along leaf veins (feature of tungro virus disease);\n"
                            f"3. Whether lesions have yellow halos (feature of bacterial blight);\n"
                            f"4. Please re-optimize YOLO coordinates to ensure center point, width and height accurately reflect lesion area.\n"
                            f"Output format same as before, return only JSON string."
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
                }
            ]
        })
        
        try:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": CONFIG["prediction"]["prediction_max_tokens"]
            }
            
            async with session.post(
                self.api_url,
                headers=headers,
                data=json.dumps(payload)
            ) as response:
                response.raise_for_status()
                response_data = await response.json()
                content = response_data['choices'][0]['message']['content']
                return self._parse_prediction(content)
                
        except Exception as e:
            logging.error(f"Low confidence recheck failed for {image_path}: {str(e)}")
            return initial_result
    
    def _parse_prediction(self, content: str) -> Dict:
        """Parse prediction results"""
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logging.warning(f"Failed to parse prediction result, content: {content}")
            return {
                "disease_type": "Unknown",
                "confidence": "0%",
                "reason": "Result format error",
                "yolo_annotations": []
            }
    
    def _parse_confidence(self, confidence_str: str) -> int:
        """Parse confidence string to integer"""
        try:
            return int(re.search(r'\d+', confidence_str).group())
        except (AttributeError, ValueError):
            return 0
    
    async def batch_predict(self, image_paths: List[str]) -> List[Dict]:
        """Batch predict images"""
        results = []
        # Process in batches according to max_batch_size
        for i in range(0, len(image_paths), self.max_batch_size):
            batch = image_paths[i:i+self.max_batch_size]
            async with aiohttp.ClientSession() as session:
                tasks = [self._async_predict(session, path) for path in batch]
                batch_results = await asyncio.gather(*tasks)
                results.extend(batch_results)
        return results


async def main():
    """Main function: execute model training and prediction process"""
    analyzer = RiceDiseaseAnalyzer()
    
    # Load data and split into training and validation sets
    logging.info("Starting to load image data...")
    disease_data = analyzer.load_disease_images()
    train_data, val_data = analyzer.split_train_val(disease_data)
    
    # Train model
    logging.info("Starting model training...")
    training_result = analyzer.train_model(train_data)
    if training_result:
        logging.info("Model training completed")
    
    # Prepare validation set image list
    val_images = []
    for disease, images in val_data.items():
        for img in images:
            val_images.append((img, disease))  # Keep true labels for evaluation
    
    # Batch predict validation set
    logging.info(f"Starting prediction for {len(val_images)} validation set images...")
    image_paths = [img for img, _ in val_images]
    predictions = await analyzer.batch_predict(image_paths)
    
    # Save prediction results
    results_df = pd.DataFrame({
        "image_path": image_paths,
        "true_label": [label for _, label in val_images],
        "predicted_label": [p["disease_type"] for p in predictions],
        "confidence": [p["confidence"] for p in predictions],
        "reason": [p["reason"] for p in predictions],
        "yolo_annotations": [json.dumps(p["yolo_annotations"], ensure_ascii=False) for p in predictions]
    })
    
    results_path = os.path.join(CONFIG["paths"]["result_dir"], "prediction_results.csv")
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    logging.info(f"Prediction results saved to {results_path}")
    
    # Save YOLO format labels
    for img_path, pred in zip(image_paths, predictions):
        if pred["yolo_annotations"]:
            # Generate txt file with same name as image
            txt_path = os.path.splitext(img_path)[0] + ".txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(pred["yolo_annotations"]))
    logging.info("YOLO format labels saved")


if __name__ == "__main__":
    asyncio.run(main())
