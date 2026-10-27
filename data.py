import torch
from torch.utils.data import Dataset
from config import label_id

def read_data(path):
    data_list = []
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
            label = label_id.get(code, -1)
            if label == -1:
                continue
            data_list.append({"text": news_text, "label": label})
    return data_list

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
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }