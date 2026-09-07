import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import WakeupDataset       # 自定义数据集类
from model import WakeupLSTM            # 双向LSTM模型
from config import BATCH_SIZE, EPOCHS, LEARNING_RATE, DEVICE, MODEL_SAVE_PATH
import os
import matplotlib.pyplot as plt

def train():
    # ===================== 1. 加载训练集和验证集 =====================
    # 初始化训练集，从train_list.txt读取音频路径与标签
    train_dataset = WakeupDataset("train_list.txt")
    # 初始化验证集
    val_dataset   = WakeupDataset("val_list.txt")
    
    # 构建训练数据加载器：打乱顺序，批次读取
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    # 构建验证数据加载器：不打乱顺序
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ===================== 2. 初始化模型、损失函数、优化器 =====================
    # 构建LSTM模型并迁移到指定设备（CPU/GPU）
    model = WakeupLSTM().to(DEVICE)
    
    # 定义损失函数：交叉熵损失，适用于二分类任务
    criterion = nn.CrossEntropyLoss()
    
    # 定义优化器：AdamW，收敛稳定且带权重衰减防止过拟合
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # 学习率调度器：验证集损失不再下降时自动降低学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)

    # ===================== 3. 初始化训练记录变量 =====================
    best_val_acc = 0.0  # 记录最高验证准确率
    history = {         # 记录每一轮的损失和准确率，用于绘图
        'train_loss': [], 
        'train_acc': [], 
        'val_loss': [], 
        'val_acc': []
    }

    # ===================== 4. 开始循环训练 =====================
    for epoch in range(EPOCHS):
        # -------------------- 训练阶段 --------------------
        model.train()  # 切换为训练模式（启用Dropout、BatchNorm等）
        train_loss = 0.0        # 累计训练损失
        train_correct = 0       # 累计预测正确数量
        train_total = 0         # 累计总样本数量

        # 遍历训练集的每一个批次
        for batch_idx, (mfcc, labels) in enumerate(train_loader):
            # 将数据移至指定设备
            mfcc, labels = mfcc.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()  # 清空上一轮梯度
            outputs = model(mfcc)  # 前向传播，得到模型输出
            loss = criterion(outputs, labels)  # 计算当前批次损失
            
            loss.backward()        # 反向传播，计算梯度
            optimizer.step()       # 更新模型参数

            # 累计损失值
            train_loss += loss.item()
            
            # 取预测概率最大的类别作为预测结果
            _, preds = torch.max(outputs, 1)
            
            # 统计正确预测数量与总数量
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            # 每50个batch打印一次训练信息
            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        # 计算本轮训练集的平均损失与准确率
        train_acc = train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        # -------------------- 验证阶段 --------------------
        model.eval()  # 切换为评估模式（关闭Dropout、BatchNorm）
        val_loss = 0.0        # 累计验证损失
        val_correct = 0       # 累计验证正确数量
        val_total = 0         # 累计验证总样本数量

        # 验证阶段不计算梯度，节省显存并加速
        with torch.no_grad():
            for mfcc, labels in val_loader:
                mfcc, labels = mfcc.to(DEVICE), labels.to(DEVICE)
                outputs = model(mfcc)          # 前向传播
                loss = criterion(outputs, labels)  # 计算损失

                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        # 计算本轮验证集平均损失与准确率
        val_acc = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        # -------------------- 记录训练历史 --------------------
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

        # 打印本轮训练&验证结果
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.4f}")

        # 更新学习率（根据验证集损失）
        scheduler.step(avg_val_loss)

        # -------------------- 保存最优模型 --------------------
        # 如果当前验证准确率高于历史最高，则保存模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> 保存最佳模型 (验证准确率 {val_acc:.4f})")

    # 训练全部结束，输出最佳结果
    print(f"训练完成，最佳验证准确率: {best_val_acc:.4f}")

    # ===================== 5. 绘制训练曲线 =====================
    plt.figure(figsize=(12, 4))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss Curve')
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.legend()
    plt.title('Accuracy Curve')
    
    # 保存图片并显示
    plt.savefig('training_curves_lstm.png')
    plt.show()
    print("训练曲线已保存为 training_curves_lstm.png")

if __name__ == "__main__":
    # 自动创建模型保存目录，避免报错
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    # 启动训练
    train()