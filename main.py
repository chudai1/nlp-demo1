import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import swanlab

import config
from data import read_data, TextDataset
from trainer import train_and_evaluate_seed

if __name__ == "__main__":

    swanlab.login(api_key="CAklUT99gTkPYC4xg1A08")
    print(f"当前设备：{config.device}")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    train_raw = read_data(config.train_path)
    dev_raw = read_data(config.dev_path)
    test_raw = read_data(config.test_path)

    train_set = TextDataset(train_raw, tokenizer, config.max_length)
    dev_set = TextDataset(dev_raw, tokenizer, config.max_length)
    test_set = TextDataset(test_raw, tokenizer, config.max_length)

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_set, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False)

    test_acc_records = []
    for run_idx, seed in enumerate(config.seed_list):
        print(f"\n== 第 {run_idx+1} 轮，seed = {seed} ==")
        
        # 固定随机种子
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        swanlab.init(
            project="demo1", 
            experiment_name=f"bert_seed_{seed}",
            config={"lr": config.lr, "batch_size": config.batch_size}
        )

        test_acc = train_and_evaluate_seed(seed, train_loader, dev_loader, test_loader)
        test_acc_records.append(test_acc)
        
        swanlab.finish()
        torch.cuda.empty_cache()

    print("\n== 实验汇总 ==")
    print(f"平均 Test Acc: {np.mean(test_acc_records)*100:.2f}% ± {np.std(test_acc_records)*100:.2f}%")