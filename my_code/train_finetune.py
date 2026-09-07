import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import WakeupDataset
from model import WakeupLSTM
from config import BATCH_SIZE, LEARNING_RATE, DEVICE, MODEL_SAVE_PATH
import os

# 微调参数
FINETUNE_EPOCHS = 10          # 微调轮数
FINETUNE_LR = 1e-4            # 更小的学习率

def fine_tune():
    # 加载数据集
    train_dataset = WakeupDataset("train_list.txt")
    val_dataset   = WakeupDataset("val_list.txt")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 创建模型并加载预训练权重
    model = WakeupLSTM().to(DEVICE)
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
        print(f"成功加载预训练模型: {MODEL_SAVE_PATH}")
    else:
        print("未找到预训练模型，将从头训练")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=FINETUNE_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)
    
    best_val_acc = 0.0
    for epoch in range(FINETUNE_EPOCHS):
        # 训练
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for mfcc, labels in train_loader:
            mfcc, labels = mfcc.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(mfcc)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        
        train_acc = train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            for mfcc, labels in val_loader:
                mfcc, labels = mfcc.to(DEVICE), labels.to(DEVICE)
                outputs = model(mfcc)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"微调 Epoch {epoch+1}/{FINETUNE_EPOCHS} | Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.4f}")
        
        scheduler.step(avg_val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> 保存最佳微调模型 (验证准确率 {val_acc:.4f})")
    
    print(f"微调完成，最佳验证准确率: {best_val_acc:.4f}")

if __name__ == "__main__":
    fine_tune()