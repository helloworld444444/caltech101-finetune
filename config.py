# config.py
import torch
import numpy as np

class Config:
    data_dir = './Caltech101'
    num_classes = 101
    train_samples_per_class = 30
    
    search_space = {
        'batch_size': [32,64],
        'lr_pretrained': [0.0009,0.001],
        'lr_new': [ 0.05, 0.06,0.07, 0.08],
        'use_pretrained': [True, False]
    }
    use_gpu = True
    train_ratio = 0.8
    step_size = 7  # 每7个epoch衰减一次
    gamma = 0.1
    num_workers = 0
    num_epochs = 50
    momentum = 0.8
    weight_decay = 1e-4
    log_dir = './runs'
    seed = 2025

# 设置随机种子
torch.manual_seed(Config.seed)
np.random.seed(Config.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(Config.seed)