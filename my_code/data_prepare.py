import os
import random
from sklearn.model_selection import train_test_split
from config import POSITIVE_DIR, NEGATIVE_DIR, HARD_NEGATIVE_DIR

# 难负例文件夹（新增）
train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2
random.seed(42)

def get_speaker_id(filename):
    import re
    basename = os.path.basename(filename)
    match = re.search(r'(\d{9})', basename)
    if match:
        return match.group(1)
    else:
        return basename.split('.')[0]

def split_by_speaker(file_list, train_ratio, val_ratio, test_ratio):
    speaker_to_files = {}
    for f in file_list:
        spk = get_speaker_id(f)
        speaker_to_files.setdefault(spk, []).append(f)
    speakers = list(speaker_to_files.keys())
    train_speakers, temp_speakers = train_test_split(speakers, train_size=train_ratio, random_state=42)
    val_speakers, test_speakers = train_test_split(temp_speakers, train_size=val_ratio/(val_ratio+test_ratio), random_state=42)
    train_files = [f for spk in train_speakers for f in speaker_to_files[spk]]
    val_files   = [f for spk in val_speakers   for f in speaker_to_files[spk]]
    test_files  = [f for spk in test_speakers  for f in speaker_to_files[spk]]
    return train_files, val_files, test_files

def main():
    if not POSITIVE_DIR.is_dir() or not NEGATIVE_DIR.is_dir():
        raise FileNotFoundError(
            "Dataset directories are missing. See README.md or set "
            "WAKEUP_POSITIVE_DIR and WAKEUP_NEGATIVE_DIR."
        )
    # 正例
    pos_files = [os.path.join(POSITIVE_DIR, f) for f in os.listdir(POSITIVE_DIR) if f.endswith('.wav')]
    
    # 原始负例
    neg_files = [os.path.join(NEGATIVE_DIR, f) for f in os.listdir(NEGATIVE_DIR) if f.endswith('.wav')]
    
    # 难负例（如果文件夹存在且有文件）
    hard_neg_files = []
    if HARD_NEGATIVE_DIR.exists():
        hard_neg_files = [str(HARD_NEGATIVE_DIR / f) for f in os.listdir(HARD_NEGATIVE_DIR) if f.endswith('.wav')]
        print(f"找到难负例 {len(hard_neg_files)} 个")
    
    # 合并负例
    all_neg_files = neg_files + hard_neg_files
    print(f"正例总数: {len(pos_files)}, 负例总数: {len(all_neg_files)}")
    
    # 分别划分正负例（说话人无关）
    pos_train, pos_val, pos_test = split_by_speaker(pos_files, train_ratio, val_ratio, test_ratio)
    neg_train, neg_val, neg_test = split_by_speaker(all_neg_files, train_ratio, val_ratio, test_ratio)
    
    train_files = pos_train + neg_train
    val_files   = pos_val   + neg_val
    test_files  = pos_test  + neg_test
    random.shuffle(train_files)
    random.shuffle(val_files)
    random.shuffle(test_files)
    
    def save_list(file_list, label_list, out_path):
        with open(out_path, 'w', encoding='utf-8') as f:
            for path, lbl in zip(file_list, label_list):
                f.write(f"{path}\t{lbl}\n")
    
    train_labels = [1 if p in pos_train else 0 for p in train_files]
    val_labels   = [1 if p in pos_val   else 0 for p in val_files]
    test_labels  = [1 if p in pos_test  else 0 for p in test_files]
    
    save_list(train_files, train_labels, "train_list.txt")
    save_list(val_files,   val_labels,   "val_list.txt")
    save_list(test_files,  test_labels,  "test_list.txt")
    
    print(f"训练集: {len(train_files)} (正{sum(train_labels)} 负{len(train_files)-sum(train_labels)})")
    print(f"验证集: {len(val_files)} (正{sum(val_labels)} 负{len(val_files)-sum(val_labels)})")
    print(f"测试集: {len(test_files)} (正{sum(test_labels)} 负{len(test_files)-sum(test_labels)})")

if __name__ == "__main__":
    main()
