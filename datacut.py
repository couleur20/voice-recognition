#!/usr/bin/env python3
"""
基于 VAD 语音起点的音频切分程序（防止段间重叠）
从每个语音段的起点开始，向后截取固定长度（默认 2 秒）。
如果与下一段起点重叠，则重叠区域填充静音（0）。
输出文件：out/学号_n.wav，每个文件长度固定为 segment_duration 秒。
"""

import os
import numpy as np
import webrtcvad
import soundfile as sf
from scipy import signal

def resample_audio(audio, orig_sr, target_sr=16000):
    if orig_sr == target_sr:
        return audio
    num_samples = int(len(audio) * target_sr / orig_sr)
    return signal.resample(audio, num_samples).astype(audio.dtype)

def convert_to_int16(audio, orig_dtype):
    if orig_dtype == np.int16:
        return audio
    elif orig_dtype == np.int32:
        return (audio >> 16).astype(np.int16)
    elif orig_dtype in (np.float32, np.float64):
        audio = np.clip(audio, -1.0, 1.0)
        return (audio * 32767).astype(np.int16)
    else:
        raise ValueError(f"不支持的音频数据类型: {orig_dtype}")

def main():
    input_file = "in.wav"
    output_dir = "out"
    os.makedirs(output_dir, exist_ok=True)

    vad_mode = 3
    frame_duration_ms = 30
    segment_duration = 2.0      # 固定切割长度（秒）

    # 1. 读取并预处理
    audio, sr = sf.read(input_file)
    orig_dtype = audio.dtype
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    audio = convert_to_int16(audio, orig_dtype)

    target_sr = 16000
    if sr != target_sr:
        audio = resample_audio(audio, sr, target_sr)
        sr = target_sr

    # 2. 整帧对齐
    frame_samples = int(sr * frame_duration_ms / 1000)
    valid_frames = len(audio) // frame_samples
    audio = audio[:valid_frames * frame_samples]
    total_samples = len(audio)

    # 3. 检测语音起点（静音后的第一个语音帧）
    vad = webrtcvad.Vad(vad_mode)
    speech_start_samples = []
    in_speech = False

    for i in range(valid_frames):
        start = i * frame_samples
        frame_bytes = audio[start:start+frame_samples].tobytes()
        is_speech = vad.is_speech(frame_bytes, sr)
        if is_speech:
            if not in_speech:
                speech_start_samples.append(start)
                in_speech = True
        else:
            in_speech = False

    if not speech_start_samples:
        print("未检测到任何语音段，退出。")
        return

    # 打印所有起点
    print(f"检测到 {len(speech_start_samples)} 个语音段起点：")
    for idx, start in enumerate(speech_start_samples, start=1):
        time_ms = start * 1000.0 / sr
        print(f"  段 {idx:3d}: 采样点 {start:7d}  (时间 {time_ms:.2f} ms)")

    # 4. 切分，处理重叠（将重叠部分补零）
    segment_samples = int(segment_duration * sr)
    file_count = 1

    for i, start in enumerate(speech_start_samples):
        # 当前段的目标结束位置（期望长度）
        ideal_end = start + segment_samples
        # 确定有效音频的结束位置：不能超过总长度，也不能超过下一段的起点
        next_start = speech_start_samples[i+1] if i+1 < len(speech_start_samples) else None
        if next_start is not None and ideal_end > next_start:
            # 重叠：有效音频只到 next_start，之后补零
            valid_end = next_start
        else:
            # 不重叠或无下一段：有效音频最多到 ideal_end 或文件末尾
            valid_end = min(ideal_end, total_samples)

        # 创建全零数组（长度为 segment_samples）
        segment = np.zeros(segment_samples, dtype=audio.dtype)
        # 计算实际可复制的长度
        copy_len = min(valid_end - start, segment_samples)
        if copy_len > 0:
            segment[:copy_len] = audio[start:start+copy_len]
        # 如果 copy_len < segment_samples，剩余部分已经是零（自动补零）

        out_path = os.path.join(output_dir, f"test{file_count:03d}.wav")
        sf.write(out_path, segment, sr, subtype='PCM_16')
        print(f"保存 {out_path}  起点 {start}，有效音频 {copy_len} 个采样点，文件固定 {segment_duration} 秒")
        file_count += 1

    print(f"\n切分完成！共生成 {file_count-1} 个文件，保存在 '{output_dir}' 文件夹中。")
    print("重叠部分已自动补零（静音）。")

if __name__ == "__main__":
    main()