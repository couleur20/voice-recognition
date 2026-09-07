import torch
import torch.nn as nn
from config import N_MFCC

class WakeupLSTM(nn.Module):
    def __init__(self, input_dim=N_MFCC, hidden_dim=128, num_layers=2, num_classes=2, dropout=0.38):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 64),  # *2 because bidirectional
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch, time, features)
        out, (hn, cn) = self.lstm(x)  # out: (batch, time, hidden*2)
        # 取最后一个时间步的输出
        out = out[:, -1, :]            # (batch, hidden*2)
        out = self.fc(out)
        return out