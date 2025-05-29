# data.py
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
import numpy as np
from collections import defaultdict
from config import Config


def get_transforms(train=True):
    return transforms.Compose([
        transforms.RandomResizedCrop(224) if train else transforms.Resize(256),
        transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
        transforms.CenterCrop(224) if not train else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def prepare_datasets():
    full_dataset = datasets.ImageFolder(Config.data_dir)

    class_indices = defaultdict(list)
    for idx, (path, label) in enumerate(full_dataset.samples):
        class_indices[label].append(idx)

    train_idx, test_idx = [], []
    for label, indices in class_indices.items():
        # 按比例划分（默认8:2），类似模型2
        split = int(len(indices) * Config.train_ratio)
        if split == 0:
            raise ValueError(f"类别 {label} 样本不足，无法按比例划分")

        np.random.shuffle(indices)
        train_idx.extend(indices[:split])
        test_idx.extend(indices[split:])

    train_dataset = Subset(datasets.ImageFolder(Config.data_dir, transform=get_transforms(True)), train_idx)
    test_dataset = Subset(datasets.ImageFolder(Config.data_dir, transform=get_transforms(False)), test_idx)

    print(f"数据集划分完成：训练集 {len(train_dataset)} 张，测试集 {len(test_dataset)} 张")
    return train_dataset, test_dataset


def create_loaders(train_set, test_set, batch_size):
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.num_workers
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.num_workers
    )
    return train_loader, test_loader
