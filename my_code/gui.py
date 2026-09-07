import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sounddevice as sd
import numpy as np
import threading
import time
import os
import librosa
import torch
from collections import deque
from model import WakeupLSTM
from config import SAMPLE_RATE, DURATION, N_MFCC, N_FFT, HOP_LEN, DEVICE, MODEL_SAVE_PATH

plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
# ---------- 全局加载模型 ----------
model = None
def load_model():
    global model
    model = WakeupLSTM().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval()

class WakeupGUI:
    def __init__(self, root):
        self.root = root
        root.title("语音唤醒词识别系统 - 小音小音")
        root.geometry("1000x750")
        
        # 学号姓名（请修改为实际信息）
        tk.Label(root, text="Wake-word recognition demo", font=("Arial", 12)).pack(pady=5)
        
        # 波形显示区
        fig_frame = tk.Frame(root)
        fig_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.fig, self.ax = plt.subplots(figsize=(9, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=fig_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 按钮区
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)
        btns = [
            ("打开文件", self.open_file),
            ("录音并识别", self.record_and_recognize),
            ("播放音频", self.play_audio),
            ("识别当前", self.recognize),
            ("批量测试", self.batch_test_gui),
            ("实时检测模式", self.toggle_realtime)
        ]
        for i, (text, cmd) in enumerate(btns):
            tk.Button(btn_frame, text=text, command=cmd, width=12).grid(row=0, column=i, padx=5)
        
        # 参数调节面板
        param_frame = tk.LabelFrame(root, text="算法参数", padx=5, pady=5)
        param_frame.pack(pady=5, fill=tk.X)
        tk.Label(param_frame, text="MFCC维数:").grid(row=0, column=0, padx=5)
        self.mfcc_var = tk.IntVar(value=N_MFCC)
        tk.Spinbox(param_frame, from_=N_MFCC, to=N_MFCC, textvariable=self.mfcc_var, width=5, state='readonly').grid(row=0, column=1)
        
        tk.Label(param_frame, text="唤醒阈值(0~1):").grid(row=0, column=2, padx=5)
        self.thresh_var = tk.DoubleVar(value=0.6)
        self.thresh_slider = tk.Scale(param_frame, from_=0.0, to=1.0, resolution=0.01,
                                      orient=tk.HORIZONTAL, variable=self.thresh_var, length=200)
        self.thresh_slider.grid(row=0, column=3, padx=5)
        
        # 结果显示
        self.result_var = tk.StringVar(value="识别结果：")
        tk.Label(root, textvariable=self.result_var, font=("Arial", 12, "bold"), fg="blue").pack(pady=5)
        
        # 日志框
        log_frame = tk.LabelFrame(root, text="日志信息")
        log_frame.pack(fill=tk.BOTH, padx=10, pady=5, expand=True)
        self.log_text = tk.Text(log_frame, height=8, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # 内部变量
        self.current_audio = None          # 当前音频数据 (1D numpy)
        self.current_path = None           # 当前文件路径
        self.recording_flag = False        # 防止重复录音
        self.temp_recording_path = "temp_recording.wav"
        
        # 实时检测相关
        self.realtime_active = False
        self.audio_buffer = deque(maxlen=int(SAMPLE_RATE * DURATION))  # 2秒环形缓冲
        self.stream = None
        self.update_interval = 0.2         # 检测间隔200ms
        
        load_model()
        self.log("系统就绪，模型加载完成。")
    
    def log(self, msg):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
    
    def plot_waveform(self, audio_data, sr=SAMPLE_RATE):
        """绘制整个波形"""
        self.ax.clear()
        times = np.linspace(0, len(audio_data)/sr, len(audio_data))
        self.ax.plot(times, audio_data, color='blue')
        self.ax.set_xlabel("时间 (秒)")
        self.ax.set_ylabel("幅度")
        self.ax.set_title("语音波形")
        self.canvas.draw()
    
    def plot_waveform_segment(self, audio_data, sr=SAMPLE_RATE, duration=1.0):
        """绘制最后一段波形（用于实时显示，duration为显示时长）"""
        if len(audio_data) == 0:
            return
        show_len = min(len(audio_data), int(sr * duration))
        segment = audio_data[-show_len:]
        times = np.linspace(-len(segment)/sr, 0, len(segment))
        self.ax.clear()
        self.ax.plot(times, segment, color='blue')
        self.ax.set_xlabel("时间 (秒)")
        self.ax.set_ylabel("幅度")
        self.ax.set_title("实时语音波形（最近{}秒）".format(duration))
        self.canvas.draw()
    
    # ---------- 文件操作 ----------
    def open_file(self):
        if self.realtime_active:
            self.toggle_realtime()
        path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if not path:
            return
        y, sr = librosa.load(path, sr=SAMPLE_RATE)
        self.current_audio = y
        self.current_path = path
        self.plot_waveform(y, sr)
        self.log(f"已加载文件：{os.path.basename(path)}")
    
    def play_audio(self):
        if self.current_audio is not None:
            sd.play(self.current_audio, SAMPLE_RATE)
            self.log("播放中...")
        else:
            self.log("没有音频可播放，请先打开文件或录音。")
    
    # ---------- 识别核心 ----------
    def predict_from_audio(self, audio):
        """给定音频数组，返回 (label, confidence) ，label=1表示唤醒词"""
        # 统一长度到2秒
        target_len = int(SAMPLE_RATE * DURATION)
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]
        # 提取MFCC，必须与训练时参数一致
        mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=self.mfcc_var.get(),
                                    n_fft=N_FFT, hop_length=HOP_LEN).T
        # 转为tensor
        mfcc_tensor = torch.from_numpy(mfcc).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(mfcc_tensor)
            prob = torch.softmax(output, dim=1).cpu().numpy()[0]
        # prob[0]: 未知, prob[1]: 唤醒
        threshold = self.thresh_var.get()
        pred = 1 if prob[1] > threshold else 0
        conf = prob[pred]
        return pred, conf
    
    def recognize(self):
        if self.current_audio is None:
            self.log("没有可识别的音频，请先打开文件或录音。")
            return
        pred, conf = self.predict_from_audio(self.current_audio)
        label = "小音小音" if pred == 1 else "未知"
        self.result_var.set(f"识别结果：{label}  (置信度: {conf:.3f})")
        self.log(f"识别完成 -> {label}，置信度 {conf:.3f}")
    
    # ---------- 单次录音 + 识别 ----------
    def record_and_recognize(self):
        if self.realtime_active:
            self.toggle_realtime()
        if self.recording_flag:
            self.log("正在录音中，请稍后...")
            return
        self.recording_flag = True
        threading.Thread(target=self._record_thread, daemon=True).start()
        self.log("开始录音（2秒），请说出唤醒词...")
    
    def _record_thread(self):
        duration = DURATION
        recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                           channels=1, dtype='float32')
        sd.wait()
        audio = recording.flatten()
        # 保存到临时文件
        import wave
        with wave.open(self.temp_recording_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes((audio * 32767).astype(np.int16).tobytes())
        # 更新当前音频
        self.current_audio = audio
        self.current_path = self.temp_recording_path
        # 主线程更新UI
        self.root.after(0, lambda: self.plot_waveform(audio, SAMPLE_RATE))
        self.root.after(0, lambda: self.log("录音结束，正在识别..."))
        # 自动识别
        pred, conf = self.predict_from_audio(audio)
        label = "小音小音" if pred == 1 else "未知"
        self.root.after(0, lambda: self.result_var.set(f"识别结果：{label}  (置信度: {conf:.3f})"))
        self.root.after(0, lambda: self.log(f"录音识别 -> {label}，置信度 {conf:.3f}"))
        self.recording_flag = False
    
    # ---------- 批量测试 ----------
    def batch_test_gui(self):
        if self.realtime_active:
            self.toggle_realtime()
        folder = filedialog.askdirectory(title="选择批量测试文件夹")
        if not folder:
            return
        output = "batch_results.txt"
        # 调用 batch_test 模块中的函数
        from batch_test import batch_test
        batch_test(folder, output)
        self.log(f"批量测试完成，结果保存在 {output}")
    
    # ---------- 实时检测模式 ----------
    def toggle_realtime(self):
        if not self.realtime_active:
            self._start_realtime()
        else:
            self._stop_realtime()
    
    def _start_realtime(self):
        if self.stream is not None:
            self._stop_realtime()
        self.realtime_active = True
        self.audio_buffer.clear()
        # 打开音频流
        self.stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32',
                                     callback=self._audio_callback, blocksize=int(SAMPLE_RATE * 0.1))
        self.stream.start()
        # 启动周期性检测
        self.root.after(int(self.update_interval * 1000), self._realtime_process)
        self.log("实时检测模式已启动（每0.2秒检测一次，延迟<200ms）")
        self.result_var.set("识别结果：实时监听中...")
    
    def _audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.audio_buffer.extend(indata.flatten())
    
    def _realtime_process(self):
        if not self.realtime_active:
            return
        # 确保缓冲区有足够数据（至少2秒）
        if len(self.audio_buffer) >= int(SAMPLE_RATE * DURATION):
            # 取最近2秒数据
            audio = np.array(list(self.audio_buffer))
            # 实时显示波形（显示最近1秒）
            self.root.after(0, lambda: self.plot_waveform_segment(audio, SAMPLE_RATE, duration=1.0))
            # 预测
            pred, conf = self.predict_from_audio(audio)
            if pred == 1:
                result_text = f"唤醒！小音小音 (置信度:{conf:.3f})"
                self.root.after(0, lambda: self.result_var.set(f"识别结果：{result_text}"))
                self.root.after(0, lambda: self.log(f"实时唤醒：{result_text}"))
            else:
                self.root.after(0, lambda: self.result_var.set(f"识别结果：未知 (置信度:{conf:.3f})"))
        # 继续下一次检测
        self.root.after(int(self.update_interval * 1000), self._realtime_process)
    
    def _stop_realtime(self):
        self.realtime_active = False
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.log("实时检测模式已停止")
        # 恢复默认波形显示区域
        if self.current_audio is not None:
            self.plot_waveform(self.current_audio, SAMPLE_RATE)

if __name__ == "__main__":
    root = tk.Tk()
    app = WakeupGUI(root)
    root.mainloop()
