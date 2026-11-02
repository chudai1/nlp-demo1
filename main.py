import numpy as np
import torch
import gc
import sys
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from functools import partial
import swanlab
import config
from data import read_data, TextDataset, collate_fn
from trainer import train_and_evaluate_seed
from utils import set_seed
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

if __name__ == "__main__":
    swanlab.login(api_key="      ")
    print(f"当前设备：{config.device}")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    train_raw, label_map, num_classes = read_data(config.train_path)
    dev_raw, _, _ = read_data(config.dev_path)
    test_raw, _, _ = read_data(config.test_path)
    
    print(f"训练集: {len(train_raw)} 条 | 验证集: {len(dev_raw)} 条 | 测试集: {len(test_raw)} 条")
    print(f"数据集中自动检测到 {num_classes} 个类别。")

    train_set = TextDataset(train_raw, tokenizer, config.max_length)
    dev_set = TextDataset(dev_raw, tokenizer, config.max_length)
    test_set = TextDataset(test_raw, tokenizer, config.max_length)

    my_collate = partial(collate_fn, tokenizer=tokenizer)

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, collate_fn=my_collate, num_workers=0)
    dev_loader = DataLoader(dev_set, batch_size=config.batch_size, shuffle=False, collate_fn=my_collate, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, collate_fn=my_collate, num_workers=0)

    test_acc_records = []
    test_f1_records = []
    test_p_records = []
    test_r_records = []
    
    for run_idx, seed in enumerate(config.seed_list):
        print(f"\n===== 第 {run_idx+1} 轮，seed = {seed} =====")
        set_seed(seed)
        swanlab.init(
            project="demo1", 
            experiment_name=f"seed{seed}_lr{config.lr}_bs{config.batch_size}_dp{config.dropout}",
            config={
                "lr": config.lr, "batch_size": config.batch_size, 
                "dropout": config.dropout, "max_norm": config.max_norm
            },
            reinit=True   
        )
        
        test_acc, test_f1, test_p, test_r = train_and_evaluate_seed(seed, train_loader, dev_loader, test_loader, num_classes)
        print(f" 函数正常返回！收到 seed {seed} 结果：acc={test_acc:.4f}, f1={test_f1:.4f}, p={test_p:.4f}, r={test_r:.4f}")
        test_acc_records.append(test_acc)
        test_f1_records.append(test_f1)
        test_p_records.append(test_p)
        test_r_records.append(test_r)

        torch.cuda.empty_cache()
        gc.collect()

    swanlab.finish()

    print("\n===== 全部种子实验汇总结果 =====")
    test_acc_arr = np.array(test_acc_records)
    test_f1_arr = np.array(test_f1_records)
    test_p_arr = np.array(test_p_records)
    test_r_arr = np.array(test_r_records)
    print(f"平均 Test Acc: {test_acc_arr.mean()*100:.2f}% ± {test_acc_arr.std()*100:.2f}%")
    print(f"平均 Test F1:  {test_f1_arr.mean()*100:.2f}% ± {test_f1_arr.std()*100:.2f}%")
    print(f"平均 Test P:   {test_p_arr.mean()*100:.2f}% ± {test_p_arr.std()*100:.2f}%")
    print(f"平均 Test R:   {test_r_arr.mean()*100:.2f}% ± {test_r_arr.std()*100:.2f}%")
