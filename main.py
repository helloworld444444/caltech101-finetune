# main.py
import os
import json
import time
from itertools import product
import pandas as pd
from data import prepare_datasets, create_loaders
from train import train_single_experiment
from config import Config
import torch


def run_experiments():
    print("\n" + "=" * 40)
    print(f"当前设备: {'GPU ✅' if torch.cuda.is_available() else 'CPU ⚠'}")
    print(f"配置选择: {'使用GPU' if Config.use_gpu else '强制使用CPU'}")
    print("=" * 40 + "\n")
    train_set, test_set = prepare_datasets()
    results = []

    for params in product(*Config.search_space.values()):
        hparams = dict(zip(Config.search_space.keys(), params))
        train_loader, test_loader = create_loaders(train_set, test_set, hparams['batch_size'])

        print(f"\n=== 训练配置: {hparams} ===")
        try:
            start = time.time()
            best_acc = train_single_experiment(hparams, train_loader, test_loader)
            results.append({
                'hparams': hparams,
                'best_acc': best_acc,
                'time': time.time() - start
            })
        except Exception as e:
            print(f"训练失败: {str(e)}")

    # 保存结果
    with open(os.path.join(Config.log_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # 结果分析
    df = pd.DataFrame(results)
    print("\n=== 最佳配置 ===")
    print(df.loc[df['best_acc'].idxmax()])


if __name__ == "__main__":
    os.makedirs(Config.log_dir, exist_ok=True)
    run_experiments()