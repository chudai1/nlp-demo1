import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

def read_data(path):
    data_list = []
    unique_codes = set()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            part_data = line.split("_!_")
            if len(part_data) == 5:
                unique_codes.add(part_data[1])

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            part_data = line.split("_!_")
            if len(part_data) == 5:
                data_list.append({"text": part_data[3], "code": part_data[1]})

    return data_list, unique_codes


class TextDataset(Dataset):
    def __init__(self, data_list, tokenizer=None, max_length=None):
        self.texts = [item["text"] for item in data_list]
        self.labels = [item["label"] for item in data_list]
        self.tokenizer = tokenizer if tokenizer is not None else BertTokenizer.from_pretrained("bert-base-chinese")
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "text": self.texts[idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

    def collate_fn(self, batch):
        texts = [item["text"] for item in batch]
        labels = torch.stack([item["labels"] for item in batch])
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        encodings["labels"] = labels
        return encodings

    def get_data_loader(self, batch_size=16, shuffle=True, **kwargs):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            **kwargs,
        )
