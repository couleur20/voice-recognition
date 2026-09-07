# Voice Wake-Word Recognition

这是一个基于 MFCC + 双向 LSTM（BiLSTM）的中文语音唤醒词二分类项目。模型将固定长度的 16 kHz、2 秒音频分为“唤醒词”和“非唤醒词”两类，并提供训练、评估、批量测试和 Tkinter 图形界面。

## 项目结构

- `my_code/data_prepare.py`：按说话人划分训练/验证/测试清单
- `my_code/dataset.py`：音频读取、补零/截断和 MFCC 特征提取
- `my_code/model.py`：双向 LSTM 分类器
- `my_code/train.py`、`my_code/test.py`：训练和测试
- `my_code/batch_test.py`：批量识别 WAV 文件
- `my_code/gui.py`：桌面演示界面
- `datacut.py`：使用 WebRTC VAD 切分音频

## 环境

建议使用 Python 3.10 或 3.11：

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## 数据与模型

音频、训练清单和模型权重默认被 `.gitignore` 排除。请将本地数据放入以下目录，或设置环境变量：

```text
positive_6675/       # 唤醒词
negative_9136/       # 普通语音/非唤醒词
hard_negative/       # 可选：困难负样本
NOISE92X_16K/        # 可选：噪声增强素材
```

环境变量：`WAKEUP_POSITIVE_DIR`、`WAKEUP_NEGATIVE_DIR`、`WAKEUP_HARD_NEGATIVE_DIR`、`WAKEUP_MODEL_PATH`。

## 运行

在仓库根目录执行：

```bash
python -m my_code.data_prepare
python my_code/train.py
python my_code/test.py
python my_code/gui.py
python my_code/batch_test.py path/to/wav_folder
```

训练清单使用绝对路径，便于本机训练；清单不应提交到公开仓库。如果需要共享实验结果，请提交脱敏后的指标、混淆矩阵和训练曲线，而不是原始录音。

## 复现说明

当前实现的输入为 16 kHz、单声道、2 秒音频，MFCC 参数为 `n_mfcc=20`、`n_fft=512`、`hop_length=160`。报告中若出现 CNN、样本数或准确率等描述，应以实际代码运行结果为准；本仓库的主模型是 BiLSTM，不是 CNN。

## 数据与隐私

公开仓库前请确认录音者授权、噪声素材许可和课程报告中的姓名/学号是否需要脱敏。不要提交访问令牌、个人录音、原始数据清单或未经许可的第三方数据。
