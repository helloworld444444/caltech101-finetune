# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
import os
import json
from tqdm import tqdm
from model import create_model
from config import Config


def train_single_experiment(hparams, train_loader, test_loader):
    if Config.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print("▶ 正在使用GPU加速")
    elif Config.use_gpu:
        device = torch.device("cpu")
        print("⚠ 配置要求使用GPU但未检测到CUDA设备，已回退到CPU")
    else:
        device = torch.device("cpu")
        print("▶ 配置为使用CPU训练")

    model = create_model(hparams['use_pretrained']).to(device)

    # 优化器配置
    if hparams['use_pretrained']:
        param_groups = [
            {'params': model.fc.parameters(), 'lr': hparams['lr_new']},
            {'params': (p for n, p in model.named_parameters()
                        if 'fc' not in n), 'lr': hparams['lr_pretrained']}
        ]
    else:
        param_groups = [{'params': model.parameters(), 'lr': hparams['lr_new']}]

    optimizer = optim.SGD(param_groups,
                          momentum=Config.momentum,
                          weight_decay=Config.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=Config.step_size,  # 从配置中读取
        gamma=Config.gamma
    )
    criterion = nn.CrossEntropyLoss()

    # 实验日志
    experiment_name = (
        f"{'pretrain' if hparams['use_pretrained'] else 'scratch'}_"
        f"bs{hparams['batch_size']}_"
        f"lr{str(hparams['lr_new']).replace('.', '_')}_"
        f"lrp{str(hparams['lr_pretrained']).replace('.', '_')}"
    )
    log_dir = os.path.join(Config.log_dir, experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_text('hparams', json.dumps(hparams))
    writer.add_text('model', str(model))
    best_acc = 0.0
    for epoch in range(Config.num_epochs):

        # 训练阶段
        model.train()
        train_loss, total_samples = 0.0, 0

        with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{Config.num_epochs}', unit='batch') as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)
                pbar.set_postfix({'loss': loss.item()})
        scheduler.step()
        avg_train_loss = train_loss / total_samples

        # 验证阶段
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        for i, param_group in enumerate(optimizer.param_groups):
            writer.add_scalar(f'LearningRate/group_{i}', param_group['lr'], epoch)

        # 记录日志
        writer.add_scalars('Loss', {'train': avg_train_loss, 'val': val_loss}, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            model_path = os.path.join(log_dir, f'best_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f"🚀 保存最佳模型到 {model_path} (准确率: {val_acc:.2f}%)")  # 新增提示

        writer.flush()  # 强制写入日志

    writer.close()
    print(f"✅ 实验完成 | 最佳准确率: {best_acc:.2f}%")

    return best_acc


def evaluate(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        # 单循环带进度条
        for inputs, labels in tqdm(data_loader, desc='Validating', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss / len(data_loader.dataset), 100 * correct / total
