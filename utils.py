import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate(model, dataloader, device):
    model.eval()
    
    all_preds = []
    all_labels = []
    sum_loss = 0.0
    total_samples = 0 

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            l = outputs["loss"]
            batch_size = labels.size(0)
            sum_loss += l.item() * batch_size 
            total_samples += batch_size
            preds = torch.argmax(outputs["logits"], dim=1)
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    model.train() 
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    avg_loss = sum_loss / total_samples
    acc = (all_preds == all_labels).float().mean().item()
    num_classes = 15
    p_list, r_list, f1_list = [], [], []

    for c in range(num_classes):

        tp = ((all_preds == c) & (all_labels == c)).sum().float()
        fp = ((all_preds == c) & (all_labels != c)).sum().float()
        fn = ((all_preds != c) & (all_labels == c)).sum().float()
        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        f1 = 2 * (p * r) / (p + r + 1e-8)
        p_list.append(p.item())
        r_list.append(r.item())
        f1_list.append(f1.item())

    precision = sum(p_list) / num_classes
    recall = sum(r_list) / num_classes
    f1 = sum(f1_list) / num_classes

    return acc, avg_loss, precision, recall, f1
