# Caltech-101 图像分类项目

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3.7%2B-blue)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

本项目实现了在 Caltech-101 数据集上微调预训练的卷积神经网络(ResNet-18)进行图像分类。项目包含完整的训练流程、超参数搜索和性能评估。

## 项目亮点

- ✅ 使用 Caltech-101 标准数据集划分
- ✅ 修改 ResNet-18 架构以适应 101 个类别
- ✅ 支持预训练模型微调与从头训练对比
- ✅ 自动化超参数搜索
- ✅ TensorBoard 可视化训练过程
- ✅ 详细的实验报告和结果分析

## 目录结构

```
caltech101-classification/
├── config.py          # 配置文件
├── data.py            # 数据处理
├── model.py           # 模型定义
├── train.py           # 训练逻辑
├── main.py            # 主程序
├── runs/              # TensorBoard 日志
├── images/            # 结果可视化图片
└── README.md          # 使用说明
```

## 快速开始

### 环境安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/caltech101-classification.git
cd caltech101-classification

项目需要以下 Python 库：

| 库名称 | 版本 | 用途 |
|--------|------|------|
| torch | >=1.10.0 | 深度学习框架 |
| torchvision | >=0.11.0 | 计算机视觉工具 |
| numpy | >=1.21.0 | 数值计算 |
| tensorboard | >=2.7.0 | 训练可视化 |
| pandas | >=1.3.0 | 数据分析 |
| tqdm | >=4.62.0 | 进度条显示 |
| matplotlib | >=3.4.0 | 结果可视化 |


### 数据集准备

1. 下载 Caltech-101 数据集：
   ```bash
   wget https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip
   unzip caltech-101.zip -d ./Caltech101
   ```

2. 数据集结构应如下：
   ```
   Caltech101/
   ├── accordion/
   ├── airplanes/
   ├── anchor/
   └── ... (共101个类别)
   ```

### 训练模型

```bash
python main.py
```

训练过程中会自动：
1. 划分训练集和测试集（8:2比例）
2. 进行超参数搜索
3. 保存最佳模型和训练日志
4. 输出性能指标

### 使用TensorBoard查看训练过程

```bash
tensorboard --logdir=runs
```

在浏览器中打开 `http://localhost:6006` 查看训练曲线和指标。

## 配置选项

在 `config.py` 中可以修改以下配置：

```python
class Config:
    data_dir = './Caltech101'          # 数据集路径
    num_classes = 101                  # 类别数量
    
    # 超参数搜索空间
    search_space = {
        'batch_size': [64],
        'lr_pretrained': [0.0009,0.001],
        'lr_new': [0.05, 0.06,0.07, 0.08],
        'use_pretrained': [True, False]  # 是否使用预训练
    }
    
    use_gpu = True                     # 是否使用GPU
    train_ratio = 0.8                  # 训练集比例
    step_size = 7                      # 学习率衰减周期
    gamma = 0.1                        # 学习率衰减因子
    num_workers = 0                    # 数据加载线程数
    num_epochs = 50                    # 训练周期数
    momentum = 0.8                     # 优化器动量
    weight_decay = 1e-4                # 权重衰减
    log_dir = './runs'                 # 日志目录
    seed = 2025                        # 随机种子
```

## 实验报告
- 数据集介绍和分析
- 模型架构说明
- 训练过程可视化
- 超参数影响分析
- 预训练与从头训练对比
- 结果讨论和结论

## 参考资源

- Caltech-101 数据集: [官方下载链接](https://data.caltech.edu/records/mzrjq-6wc02)
