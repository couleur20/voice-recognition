import torch
import os
import librosa
import numpy as np
from model import WakeupLSTM
from config import DEVICE, SAMPLE_RATE, DURATION, N_MFCC, N_FFT, HOP_LEN, MODEL_SAVE_PATH

def predict_one(model, wav_path):
    """杩斿洖 (pred_label, confidence)"""
    y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
    target_len = int(SAMPLE_RATE * DURATION)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LEN).T
    mfcc = torch.from_numpy(mfcc).float().unsqueeze(0).to(DEVICE)  # add batch
    with torch.no_grad():
        outputs = model(mfcc)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()
    return pred, confidence

def batch_test(folder_path, output_txt):
    model = WakeupLSTM().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval()
    
    results = []
    for fname in os.listdir(folder_path):
        if not fname.endswith('.wav'):
            continue
        full_path = os.path.join(folder_path, fname)
        pred, conf = predict_one(model, full_path)
        label_str = "灏忛煶灏忛煶" if pred == 1 else "鏈煡"
        results.append(f"{fname} {label_str} (缃俊搴?{conf:.3f})")
    
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write("\n".join(results))
    print(f"鎵归噺娴嬭瘯瀹屾垚锛岀粨鏋滀繚瀛樿嚦 {output_txt}")

if __name__ == "__main__":
    # 绀轰緥锛氭祴璇曚綘鑷繁鐨勪竴涓枃浠跺す锛堟瘮濡?test_wavs 鐩綍锛?
    import sys
    if len(sys.argv) > 1:
        folder = sys.argv[1]
    else:
        folder = input("璇疯緭鍏ヨ娴嬭瘯鐨勬枃浠跺す璺緞: ")
    output_file = "batch_results.txt"
    batch_test(folder, output_file)
