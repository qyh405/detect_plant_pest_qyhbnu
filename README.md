# 水稻病害识别工具 (Rice Disease Detection Tool)

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)


## 项目简介 (Project Overview)

本项目是一款专注于水稻病害识别的工具，能够精准识别多种水稻病害（如稻瘟病、细菌性枯萎病等），并输出符合YOLO格式的病变位置坐标。通过API调用视觉模型进行训练与预测，结合二次训练机制持续优化识别精度，适用于农业病虫害监测、学术研究及农业生产辅助决策场景。

This project is a tool focused on rice disease and pest detection, capable of accurately identifying various rice diseases (such as rice blast, bacterial blight, etc.) and outputting lesion coordinates in YOLO format. It uses API calls to visual models for training and prediction, combined with a retraining mechanism to continuously optimize recognition accuracy, suitable for agricultural pest monitoring, academic research, and agricultural production decision-making assistance.

## 核心功能 (Key Features)

- 支持多种水稻病虫害的精准分类识别（可通过配置文件灵活扩展）
- 输出标准化YOLO格式的病变区域坐标（类别索引、中心点坐标、宽高）
- 自动划分训练集与验证集，提供详细的模型评估报告
- 低置信度结果二次校验及错误案例二次训练，持续提升识别精度
- 图像预处理（对比度增强、尺寸调整）与缓存机制，优化处理性能
- 完善的日志记录与结果保存，便于后续分析与追溯

- Supports accurate classification of multiple rice diseases (flexibly extensible via configuration file)
- Outputs standardized YOLO-format lesion coordinates (class index, center coordinates, width/height)
- Automatically splits training and validation sets, providing detailed model evaluation reports
- Secondary verification for low-confidence results and retraining with error cases to continuously improve accuracy
- Image preprocessing (contrast enhancement, resizing) and caching mechanism to optimize processing performance
- Comprehensive logging and result saving for subsequent analysis and traceability

## 环境要求 (Environment Requirements)

- Python 3.12 及以上版本 (Python 3.12 or higher)
- 依赖库：见 `requirements.txt`

## 安装步骤 (Installation Steps)

1. 克隆仓库到本地 (Clone the repository locally)
   ```bash
   git clone https://github.com/your-username/rice-disease-detection.git
   cd rice-disease-detection
   ```

2. 安装依赖包 (Install dependencies)
   ```bash
   pip install -r requirements.txt
   ```

3. 配置API密钥 (Configure API key)：见【配置说明】部分

## 配置说明 (Configuration)

项目通过 `config.json` 文件进行参数配置，关键配置项说明如下：

The project is configured through the `config.json` file, with key configuration items explained as follows:

### 1. API 配置 (API Settings)
```json
"api": {
    "api_key": "YOUR_API_KEY",  // 替换为你的API密钥 (Replace with your API key)
    "model": "qwen-vl-max",     // 模型名称 (Model name)
    "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",  // API接口地址 (API endpoint)
    "max_batch_size": 10        // 批量预测最大数量 (Maximum batch size for prediction)
}
```

### 2. 路径配置 (Path Settings)
```json
"paths": {
    "result_dir": "result_plant_pest/",  // 结果输出目录 (Result output directory)
    "cache_dir": "${result_dir}/.cache",  // 缓存目录（自动关联result_dir）(Cache directory, automatically associated with result_dir)
    "log_dir": "${result_dir}/.log",      // 日志目录 (Log directory)
    "data_dir": "small_test"  // 图像数据目录 (Image data directory)
}
```
> 数据目录需按病害类型创建子文件夹（如 `稻瘟病/`、`细菌性枯萎病/`），并放入对应图像

> The data directory should have subfolders by disease type (e.g., `稻瘟病/`, `细菌性枯萎病/`) with corresponding images

### 3. 图像处理 (Image Processing)
```json
"image": {
    "resized_image_size": [256, 256],  // 图像缩放尺寸 (Image resizing dimensions)
    "contrast_enhance_ratio": 1.3,     // 对比度增强比例 (Contrast enhancement ratio)
    "image_quality": 95                // 图像压缩质量 (Image compression quality)
}
```

### 4. 训练配置 (Training Settings)
```json
"training": {
    "target_diseases": ["稻瘟病", "细菌性枯萎病", "水稻东格鲁病毒病", "褐斑病"],  // 目标病害类型 (Target disease types)
    "retrain_threshold": 80,           // 二次训练精度阈值（%）(Retraining accuracy threshold (%))
    "sample_size": 200,                // 每类病害训练样本量 (Number of training samples per disease)
    "test_size": 0.05,                 // 验证集比例 (Validation set ratio)
    "random_state": 42,                // 随机种子（确保结果可复现）(Random seed for reproducibility)
    "force_retrain": true              // 是否强制重训（忽略缓存）(Force retraining, ignoring cache)
}
```

### 5. 预测配置 (Prediction Settings)
```json
"prediction": {
    "low_confidence_threshold": 70,    // 低置信度阈值（%）(Low confidence threshold (%))
    "max_tokens": 4096,                // 训练时最大tokens (Maximum tokens for training)
    "prediction_max_tokens": 1024      // 预测时最大tokens (Maximum tokens for prediction)
}
```

## 使用步骤 (Usage)

1. 准备数据：在 `data_dir` 配置的目录下，按病害类型创建子文件夹并放入对应图像

   Prepare data: In the directory configured by `data_dir`, create subfolders by disease type and place corresponding images

2. 配置API密钥：将 `config.json` 中的 `api_key` 替换为你的实际API密钥

   Configure API key: Replace `api_key` in `config.json` with your actual API key

3. 运行程序：

   Run the program:
   ```bash
   python detect_plant_pest_with_prompt_chn.py
   ```

4. 查看结果：

   View results:
   - 预测结果（含YOLO标签）保存在 `result_dir` 中 (Prediction results including YOLO labels are saved in `result_dir`)
   - 日志文件位于 `log_dir` 目录 (Log files are in `log_dir`)
   - 错误案例及二次训练结果在对应子目录 (Error cases and retraining results are in corresponding subdirectories)
   - 评估报告以Excel和JSON格式保存 (Evaluation reports are saved in Excel and JSON formats)

## 输出说明 (Output Explanation)

### YOLO格式标签 (YOLO-format Labels)
每个图像对应一个 `.txt` 文件，每行格式为：

Each image corresponds to a `.txt` file, with each line in the format:
```
类别索引 中心点x 中心点y 宽度 高度
class_index x_center y_center width height
```
- 坐标为0-1的归一化值（相对于图像宽高）(Coordinates are normalized to 0-1 relative to image width and height)
- 示例 (Example)：`0 0.35 0.42 0.2 0.18` 表示类别索引为0的病变，中心点在图像35%宽度、42%高度处，宽占20%，高占18%

### 评估报告 (Evaluation Report)
包含以下关键信息：

Contains the following key information:
- 总体准确率 (Overall accuracy)
- 各类别准确率 (Per-class accuracy)
- 每张图像的预测结果（真实标签、预测标签、置信度、判断依据）(Prediction results for each image (true label, predicted label, confidence, reasoning))
- 错误案例详情（用于二次训练）(Details of error cases (for retraining))

## 项目结构 (Project Structure)
```
rice-disease-detection/
├── detect_plant_pest_with_prompt_chn.py  # 核心程序文件 (Core program file)
├── config.json                          # 配置文件 (Configuration file)
├── config.example.json                  # 配置模板 (Configuration template)
├── requirements.txt                     # 依赖清单 (Dependency list)
├── LICENSE                              # 开源许可证 (Open source license)
├── README.md                            # 项目说明文档 (Project documentation)
├── data/                                # 图像数据目录 (Image data directory)
│   ├── 稻瘟病/                          # 稻瘟病图像 (Rice blast images)
│   ├── 细菌性枯萎病/                      # 细菌性枯萎病图像 (Bacterial blight images)
│   └── ...
└── results/                             # 结果输出目录 (Result output directory)
    ├── yolo_labels/                     # YOLO格式标签 (YOLO-format labels)
    ├── .cache/                          # 缓存文件 (Cache files)
    ├── .log/                            # 日志文件 (Log files)
    └── after_retrain/                   # 二次训练后结果 (Results after retraining)
```

## 许可证 (License)
本项目采用 [AGPLv3 License](LICENSE)开源，详情参见 LICENSE 文件。

This project is open-sourced under the [AGPLv3 License](LICENSE). See the LICENSE file for details.
