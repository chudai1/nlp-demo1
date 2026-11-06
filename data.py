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
                
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            part_data = line.split("_!_")
            if len(part_data) != 5: continue
            code = part_data[1]
            news_text = part_data[3]
            data_list.append({"text": news_text, "code": code})
            
    return data_list, unique_codes


class TextDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_length):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        return {
            "text": item["text"], 
            "label": item["label"],
            "tokenizer": self.tokenizer,  
            "max_length": self.max_length 
        }

def collate_fn(batch):

    tokenizer = batch[0]["tokenizer"]
    max_length = batch[0]["max_length"]
    
    texts = [item["text"] for item in batch]
    labels = [item["label"] for item in batch]
    
    encoding = tokenizer(
        texts,
        max_length=max_length,
        padding="longest",  
        truncation=True,
        return_tensors="pt"
    )
    
    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "label": torch.tensor(labels, dtype=torch.long)
    }
