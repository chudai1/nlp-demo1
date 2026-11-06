import os
import json
import torch

base_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(base_dir, "train_3k.txt")
dev_path = os.path.join(base_dir, "dev_1k.txt")
test_path = os.path.join(base_dir, "test_1k.txt")

with open(os.path.join(base_dir, "configs", "params.json"), "r", encoding="utf-8") as f:
    params = json.load(f)

model_name = params["model_name"]
max_length = params["max_length"]
batch_size = params["batch_size"]
lr = params["lr"]
max_epochs = params["max_epochs"]
patience = params["patience"]
seed_list = params["seed_list"]
dropout = params["dropout"]
max_norm = params["max_norm"]
warmup_ratio = params["warmup_ratio"]

output_dir = os.path.join(base_dir, "outputs")
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
