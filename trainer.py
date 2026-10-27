import os
import torch
from tqdm import tqdm
import swanlab
from transformers import get_linear_schedule_with_warmup
import config
from utils import evaluate
from model import TextClassifier

def train_and_evaluate_seed(seed, train_loader, dev_loader, test_loader):

    model = TextClassifier().to(config.device)
    
    # 优化器与调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    num_training_steps = len(train_loader) * config.max_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * num_training_steps), 
        num_training_steps=num_training_steps
    )

    best_dev_acc = 0.0
    best_model_path = os.path.join(config.dir_path, f"best_bert_seed{seed}.pth")
    no_improve = 0
    stop_flag = False


    for epoch in range(config.max_epochs):
        if stop_flag:
            break
        
        model.train()
        sum_loss = 0.0
        progress = tqdm(train_loader, desc=f"seed{seed} TRAIN {epoch+1}")
        
        for step, batch in enumerate(progress):
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            labels = batch["label"].to(config.device)

            optimizer.zero_grad()
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            l = output["loss"]
            l.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            sum_loss += l.item()
            progress.set_postfix({"loss": f"{l.item():.4f}"})
            
            if step % 10 == 0:
                swanlab.log({"train_loss": l.item()}, step=epoch * len(train_loader) + step)

        avg_train_loss = sum_loss / len(train_loader)
        dev_acc, dev_loss = evaluate(model, dev_loader, config.device)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Dev Acc: {dev_acc*100:.2f}%")
        swanlab.log({"epoch_train_loss": avg_train_loss, "epoch_dev_acc": dev_acc})



        # 早停
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            torch.save(model.state_dict(), best_model_path)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= config.patience:
                print(f"seed {seed} 触发早停！")
                stop_flag = True


    checkpoint = torch.load(best_model_path, map_location=config.device, weights_only=True)
    model.load_state_dict(checkpoint)
    test_acc, test_loss = evaluate(model, test_loader, config.device)
    print(f"seed {seed} 测试集 Acc: {test_acc*100:.2f}%")
    
    return test_acc