import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from dataset import WakeupDataset
from model import WakeupLSTM
from config import DEVICE, MODEL_SAVE_PATH, BATCH_SIZE

plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

def evaluate():
    test_dataset = WakeupDataset("test_list.txt")
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = WakeupLSTM().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval()
    
    y_true, y_pred = [], []
    with torch.no_grad():
        for mfcc, labels in test_loader:
            mfcc = mfcc.to(DEVICE)
            outputs = model(mfcc)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    
    # 分类报告
    target_names = ['未知', '小音小音']
    report = classification_report(y_true, y_pred, target_names=target_names)
    print(report)
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'混淆矩阵 (准确率: {np.trace(cm)/np.sum(cm):.4f})')
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    # 保存文本报告
    with open('classification_report.txt', 'w') as f:
        f.write(report)
    print("评估结果已保存为 classification_report.txt 和 confusion_matrix.png")

if __name__ == "__main__":
    evaluate()