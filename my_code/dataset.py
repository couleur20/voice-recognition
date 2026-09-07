import torch
from torch.utils.data import Dataset
import librosa
import numpy as np
from config import SAMPLE_RATE, DURATION, N_MFCC, N_FFT, HOP_LEN

class WakeupDataset(Dataset):
    def __init__(self, list_file):
        self.data = []
        with open(list_file, 'r', encoding='utf-8') as f:
            for line in f:
                path, label = line.strip().split('\t')
                self.data.append((path, int(label)))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        path, label = self.data[idx]
        # 加载音频，固定16kHz
        y, sr = librosa.load(path, sr=SAMPLE_RATE)
        # 确保长度至少2秒，不足则补零，超出则截断
        target_len = int(SAMPLE_RATE * DURATION)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]
        
        # 提取MFCC (delta和delta-delta可选，先只取静态MFCC)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LEN)
        # mfcc shape: (N_MFCC, time) -> 转为 (time, N_MFCC)
        mfcc = mfcc.T  # 现在 (T, N_MFCC)
        
        # 转为tensor
        mfcc = torch.from_numpy(mfcc).float()
        return mfcc, torch.tensor(label, dtype=torch.long)