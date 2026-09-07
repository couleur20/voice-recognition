import os
import shutil

from config import NEGATIVE_DIR, HARD_NEGATIVE_DIR

# 原始负例文件夹
neg_dir = str(NEGATIVE_DIR)
# 存放难负例的目标文件夹
hard_neg_dir = str(HARD_NEGATIVE_DIR)

os.makedirs(hard_neg_dir, exist_ok=True)

# 如果你的文件名包含“小”字或特定标识，可以自动筛选
# 否则建议手动复制。这里示例：文件名包含“小”或“xiao”的复制过去
count = 0
for f in os.listdir(neg_dir):
    if '小' in f or 'xiao' in f.lower():
        src = os.path.join(neg_dir, f)
        dst = os.path.join(hard_neg_dir, f)
        shutil.copy(src, dst)
        count += 1
        print(f"复制: {f}")
print(f"共复制 {count} 个难负例。")
