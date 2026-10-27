
import torch

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total_num = 0
    sum_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            l = outputs["loss"]
            sum_loss += l.item()
            
            pred = torch.argmax(outputs["logits"], dim=1)
            correct += (pred == labels).sum().item()
            total_num += labels.size(0)
    model.train()
    acc = correct / total_num
    avg_loss = sum_loss / len(dataloader)
    return acc, avg_loss