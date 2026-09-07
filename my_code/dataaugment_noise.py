import os
import random
import numpy as np
import librosa
import soundfile as sf

# ==================== 配置参数 ====================
from config import NEGATIVE_DIR, PROJECT_ROOT
DATA_DIR = str(NEGATIVE_DIR)
NOISE_DIR = str(PROJECT_ROOT / "NOISE92X_16K")
OUTPUT_DIR = str(PROJECT_ROOT / "augmented_data")

TARGET_SR = 16000                         # 目标采样率（Hz）
DURATION_SEC = 2.0                        # 固定音频长度（秒）
TARGET_SAMPLES = int(TARGET_SR * DURATION_SEC)   # 32000

# 随机选择文件数量
NUM_FILES_TO_SELECT = 50

# 噪声添加参数：信噪比范围（dB）
SNR_DB_RANGE = (5, 20)

# ==================== 辅助函数 ====================
def load_and_fix_audio(file_path):
    """加载音频，重采样至 TARGET_SR，并固定长度（截断或补零）"""
    audio, sr = librosa.load(file_path, sr=TARGET_SR)
    if len(audio) >= TARGET_SAMPLES:
        audio = audio[:TARGET_SAMPLES]
    else:
        pad_width = TARGET_SAMPLES - len(audio)
        audio = np.pad(audio, (0, pad_width), 'constant', constant_values=0)
    return audio

def add_noise_from_folder(clean_audio, noise_dir, snr_db):
    """从噪声文件夹随机选取一个噪声文件，截取与 clean_audio 等长的片段，按信噪比混合"""
    noise_files = [f for f in os.listdir(noise_dir) if f.lower().endswith('.wav')]
    if not noise_files:
        raise FileNotFoundError(f"噪声文件夹 {noise_dir} 中没有找到 .wav 文件")
    noise_file = random.choice(noise_files)
    noise_path = os.path.join(noise_dir, noise_file)
    noise_audio, sr = librosa.load(noise_path, sr=TARGET_SR)
    
    # 如果噪声比目标长，随机裁剪；否则循环填充
    if len(noise_audio) >= TARGET_SAMPLES:
        start = random.randint(0, len(noise_audio) - TARGET_SAMPLES)
        noise_segment = noise_audio[start:start + TARGET_SAMPLES]
    else:
        repeats = int(np.ceil(TARGET_SAMPLES / len(noise_audio)))
        noise_segment = np.tile(noise_audio, repeats)[:TARGET_SAMPLES]
    
    # 计算信噪比混合
    clean_rms = np.sqrt(np.mean(clean_audio ** 2))
    noise_rms = np.sqrt(np.mean(noise_segment ** 2))
    if clean_rms == 0:
        clean_rms = 1e-6
    if noise_rms == 0:
        noise_rms = 1e-6
    snr_linear = 10 ** (snr_db / 10.0)
    noise_scaling = clean_rms / (noise_rms * np.sqrt(snr_linear))
    scaled_noise = noise_segment * noise_scaling
    mixed = clean_audio + scaled_noise
    # 防止削波
    max_val = np.max(np.abs(mixed))
    if max_val > 1.0:
        mixed = mixed / max_val
    return mixed

def main():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 获取所有原始 wav 文件
    all_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith('.wav')]
    if not all_files:
        print(f"错误：在 {DATA_DIR} 中没有找到 .wav 文件")
        return
    
    # 随机选择指定数量的文件（若不足则全部选取）
    if len(all_files) <= NUM_FILES_TO_SELECT:
        selected_files = all_files
        print(f"原始文件总数 {len(all_files)} 不足 {NUM_FILES_TO_SELECT}，将处理全部文件")
    else:
        selected_files = random.sample(all_files, NUM_FILES_TO_SELECT)
        print(f"从 {len(all_files)} 个文件中随机选择了 {NUM_FILES_TO_SELECT} 个文件")
    
    print(f"输出目录: {OUTPUT_DIR}\n")
    
    total_generated = 0
    for wav_file in selected_files:
        base_name = os.path.splitext(wav_file)[0]
        input_path = os.path.join(DATA_DIR, wav_file)
        
        # 加载并标准化长度
        clean_audio = load_and_fix_audio(input_path)
        
        # 随机生成信噪比（5~20 dB）
        snr_db = random.uniform(*SNR_DB_RANGE)
        augmented = add_noise_from_folder(clean_audio, NOISE_DIR, snr_db)
        
        # 确保长度
        if len(augmented) >= TARGET_SAMPLES:
            augmented = augmented[:TARGET_SAMPLES]
        else:
            pad_width = TARGET_SAMPLES - len(augmented)
            augmented = np.pad(augmented, (0, pad_width), 'constant', constant_values=0)
        
        # 保存文件
        output_filename = f"{base_name}_noise_{int(snr_db)}dB.wav"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        sf.write(output_path, augmented, TARGET_SR)
        print(f"生成: {output_filename} (SNR={snr_db:.1f}dB)")
        total_generated += 1
    
    print(f"\n数据增强完成！共生成 {total_generated} 个文件，保存在 {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
