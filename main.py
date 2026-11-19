import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import numpy as np
import torch
import swanlab
import config
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from data import read_data, TextDataset
from trainer import train_and_evaluate_seed
from utils import set_seed

if __name__ == "__main__":
    swanlab.login(api_key="")
    print(f"当前设备：{config.device}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    train_raw, train_codes = read_data(config.train_path)
    dev_raw, dev_codes = read_data(config.dev_path)
    test_raw, test_codes = read_data(config.test_path)

    all_codes = train_codes.union(dev_codes).union(test_codes)
    sorted_codes = sorted(list(all_codes))
    label_map = {code: idx for idx, code in enumerate(sorted_codes)}
    num_classes = len(label_map)

    for item in train_raw:
        item["label"] = label_map[item["code"]]
    for item in dev_raw:
        item["label"] = label_map[item["code"]]
    for item in test_raw:
        item["label"] = label_map[item["code"]]

    print(f"训练集: {len(train_raw)} | 验证集: {len(dev_raw)} | 测试集: {len(test_raw)}")
    print(f"自动检测到 {num_classes} 个类别。")

    train_set = TextDataset(train_raw, tokenizer, config.max_length)
    dev_set = TextDataset(dev_raw, tokenizer, config.max_length)
    test_set = TextDataset(test_raw, tokenizer, config.max_length)

    dev_loader = dev_set.get_data_loader(batch_size=config.batch_size, shuffle=False)
    test_loader = test_set.get_data_loader(batch_size=config.batch_size, shuffle=False)

    all_metrics = {k: [] for k in ["acc", "macro_p", "macro_r", "macro_f1", "micro_p", "micro_r", "micro_f1"]}

    for run_idx, seed in enumerate(config.seed_list):
        print(f"\n===== 第 {run_idx+1} 轮，seed = {seed} =====")
        set_seed(seed)
        train_loader = train_set.get_data_loader(batch_size=config.batch_size, shuffle=True)

        swanlab.init(
            project="tnews_classification",
            experiment_name=f"seed{seed}_lr{config.lr}_bs{config.batch_size}",
            config={"lr": config.lr, "batch_size": config.batch_size, "dropout": config.dropout, "run_name": config.run_name},
            reinit=True,
        )

        test_metrics = train_and_evaluate_seed(seed, train_loader, dev_loader, test_loader, num_classes)
        for key in all_metrics:
            all_metrics[key].append(test_metrics[key])

    print("\n" + "=" * 20 + " 全部种子实验最终汇总结果 " + "=" * 20)

    def print_metric(name, key):
        if not all_metrics[key]:
            return
        arr = np.array(all_metrics[key])
        print(f"{name:<20} {arr.mean()*100:.2f}% ± {arr.std()*100:.2f}%")

    print_metric("平均 Test Acc:", "acc")
    print("-" * 50)
    print_metric("平均 Test Macro-F1:", "macro_f1")
    print_metric("平均 Test Micro-F1:", "micro_f1")
    print("=" * 60 + "\n")

    swanlab.finish()
