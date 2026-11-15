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


def evaluate(model, dataloader, device, num_classes):
    model.eval()
    all_preds, all_labels = [], []
    sum_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]
            sum_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

            all_preds.append(torch.argmax(outputs["logits"], dim=1).cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    avg_loss = sum_loss / total_samples
    acc = (all_preds == all_labels).float().mean().item()

    p_list, r_list, f1_list = [], [], []
    total_tp, total_fp, total_fn = 0, 0, 0

    for c in range(num_classes):
        tp = ((all_preds == c) & (all_labels == c)).sum().float()
        fp = ((all_preds == c) & (all_labels != c)).sum().float()
        fn = ((all_preds != c) & (all_labels == c)).sum().float()

        total_tp += tp
        total_fp += fp
        total_fn += fn

        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        f1 = 2 * (p * r) / (p + r + 1e-8)

        p_list.append(p.item())
        r_list.append(r.item())
        f1_list.append(f1.item())

    macro_p = sum(p_list) / num_classes
    macro_r = sum(r_list) / num_classes
    macro_f1 = sum(f1_list) / num_classes

    micro_p = total_tp / (total_tp + total_fp + 1e-8)
    micro_r = total_tp / (total_tp + total_fn + 1e-8)
    micro_f1 = 2 * (micro_p * micro_r) / (micro_p + micro_r + 1e-8)

    return {
        "acc": acc,
        "loss": avg_loss,
        "macro_p": macro_p,
        "macro_r": macro_r,
        "macro_f1": macro_f1,
        "micro_p": float(micro_p),
        "micro_r": float(micro_r),
        "micro_f1": float(micro_f1),
    }
