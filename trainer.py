import os
import torch
import gc
from tqdm import tqdm
import swanlab
from transformers import get_linear_schedule_with_warmup
import config
from utils import evaluate
from model import TextClassifier

def train_and_evaluate_seed(seed, train_loader, dev_loader, test_loader, num_classes):
    model = TextClassifier(num_labels=num_classes).to(config.device) 
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    num_training_steps = len(train_loader) * config.max_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * num_training_steps), num_training_steps=num_training_steps
    )

    best_dev_acc = -1.0 
    best_model_path = os.path.join(config.base_dir, f"best_bert_seed{seed}.pth")
    no_improve = 0
    stop_flag = False

    for epoch in range(config.max_epochs):
        if stop_flag:
            break  
        model.train()
        sum_loss = 0.0
        total_samples = 0
        progress = tqdm(train_loader, desc=f"seed{seed} TRAIN {epoch+1}")
        
        for step, batch in enumerate(progress):
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            labels = batch["label"].to(config.device)

            optimizer.zero_grad()
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            l = output["loss"]
            l.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
            optimizer.step()
            scheduler.step()

            sum_loss += l.item() * labels.size(0) 
            total_samples += labels.size(0)
            progress.set_postfix({"loss": f"{l.item():.4f}"})
            
            if step % 10 == 0:
                swanlab.log({"train_loss": l.item()}, step=epoch * len(train_loader) + step)

        avg_train_loss = sum_loss / total_samples
        dev_acc, dev_loss, dev_p, dev_r, dev_f1 = evaluate(model, dev_loader, config.device)
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Dev Acc: {dev_acc*100:.2f}% | Dev F1: {dev_f1*100:.2f}% | Dev P: {dev_p*100:.2f}% | Dev R: {dev_r*100:.2f}%")
        
        swanlab.log({
            "epoch_train_loss": avg_train_loss,
            "epoch_dev_acc": dev_acc,
            "epoch_dev_f1": dev_f1,
            "epoch_dev_loss": dev_loss,
            "epoch_dev_p": dev_p,
            "epoch_dev_r": dev_r
        })

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_dev_acc
            }
            
            torch.save(checkpoint, best_model_path)
            print(f"seed{seed} 表现更好，保存 checkpoint (Acc: {best_dev_acc*100:.2f}%)")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= config.patience:
                print(f"seed {seed} 触发早停！")
                stop_flag = True

    print("\n== 训练结束，加载最优权重做测试 ==")
    checkpoint = torch.load(best_model_path, map_location=config.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict']) 
    model.eval()
    
    print(" 权重加载成功，开始跑测试集")
    test_acc, test_loss, test_p, test_r, test_f1 = evaluate(model, test_loader, config.device)
    
    print(f"seed {seed} 测试集 -> Acc: {test_acc*100:.2f}% | F1: {test_f1*100:.2f}% | P: {test_p*100:.2f}% | R: {test_r*100:.2f}%")

    model = model.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    return test_acc, test_f1, test_p, test_r
