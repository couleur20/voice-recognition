import os
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
POSITIVE_DIR = Path(os.getenv("WAKEUP_POSITIVE_DIR", PROJECT_ROOT / "positive_6675"))
NEGATIVE_DIR = Path(os.getenv("WAKEUP_NEGATIVE_DIR", PROJECT_ROOT / "negative_9136"))
HARD_NEGATIVE_DIR = Path(os.getenv("WAKEUP_HARD_NEGATIVE_DIR", PROJECT_ROOT / "hard_negative"))
MODEL_SAVE_PATH = Path(os.getenv("WAKEUP_MODEL_PATH", PROJECT_ROOT / "my_code" / "models" / "wakeup_model.pth"))

# 音频参数
SAMPLE_RATE = 16000
DURATION = 2.0          # 固定2秒
N_MFCC = 20             # MFCC维度
N_FFT = 512
HOP_LEN = 160           # 10ms一帧

# 模型参数
INPUT_SIZE = N_MFCC
NUM_CLASSES = 2
LEARNING_RATE = 0.0005   # 降低到 0.0005
BATCH_SIZE = 64
EPOCHS = 50              # 增加到 50

# 设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
