import torch
from torch.utils.data import Dataset

def read_data(path):
    data_list = []
    unique_codes = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            part_data = line.split("_!_")
            if len(part_data) == 5:
                unique_codes.add(part_data[1])
                
    sorted_codes = sorted(list(unique_codes))
    dynamic_label_id = {code: idx for idx, code in enumerate(sorted_codes)}
    num_classes = len(dynamic_label_id)

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            part_data = line.split("_!_")
            if len(part_data) != 5: 
                continue
            code = part_data[1]
            news_text = part_data[3]
            label = dynamic_label_id.get(code, -1)
            if label == -1: 
                continue
            
            data_list.append({"text": news_text, "label": label})
            
    return data_list, dynamic_label_id, num_classes

class TextDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_length=128):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        text = item["text"]
        label = item["label"]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
        )
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "label": label
        }

def collate_fn(batch, tokenizer):

    texts = [item["input_ids"] for item in batch]
    labels = [item["label"] for item in batch]
    
    # 动态补齐
    encoding = tokenizer.pad(
        {"input_ids": texts},
        padding="longest", 
        return_tensors="pt"
    )
    
    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "label": torch.tensor(labels, dtype=torch.long)
    }
