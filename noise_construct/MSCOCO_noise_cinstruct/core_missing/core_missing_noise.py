import random
from tqdm import tqdm

# 读取原始文本数据
file_path = '/home/zbm/xjd/NPC-master/dataset/core_missing_Error_noise_coco/annotations/scan_split/0_noise_train_caps.txt'

# 读取原始文本文件
with open(file_path, 'r', encoding='utf-8') as f:
    raw_texts = f.readlines()

# 去除文本中的换行符
raw_texts = [text.strip() for text in raw_texts]

# 噪声文件路径
noise_file_path = '/home/zbm/xjd/NPC-master/dataset/core_missing_Error_noise_coco/annotations/scan_split/1.0_noise_train_caps.txt'

# 读取噪声文件
with open(noise_file_path, 'r', encoding='utf-8') as f:
    noise_texts = f.readlines()

# 去除噪声文本中的换行符
noise_texts = [text.strip() for text in noise_texts]

# 确保噪声文本数量足够
if len(noise_texts) < len(raw_texts):
    print(f"警告: 噪声文本数量({len(noise_texts)})少于原始文本数量({len(raw_texts)})")
    noise_texts = noise_texts * (len(raw_texts) // len(noise_texts) + 1)

# 多个替换比例
replace_ratios = [0.2,0.4,0.6]

for replace_ratio in replace_ratios:
    # 创建原始文本的副本
    modified_texts = raw_texts.copy()
    
    num_texts = len(modified_texts)
    num_to_replace = int(num_texts * replace_ratio)
    
    # 随机选择需要替换的文本索引
    indices_to_replace = random.sample(range(num_texts), num_to_replace)
    
    # 替换文本
    for idx in tqdm(indices_to_replace, desc=f"Replacing texts ({replace_ratio})"):
        modified_texts[idx] = noise_texts[idx]
    
    # 输出文件路径
    output_file_path = f'/home/zbm/xjd/NPC-master/dataset/core_missing_Error_noise_coco/annotations/scan_split/{replace_ratio}_noise_train_caps.txt'
    
    # 保存替换后的文本到文件
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for text in modified_texts:
            # 去除开头的编号和点（如 "1. "）
            cleaned_text = text.lstrip().split('. ', 1)
            if len(cleaned_text) == 2 and cleaned_text[0].isdigit():
                text = cleaned_text[1]
            f.write(text + "\n")
    
    print(f"{replace_ratio} 替换比例的文本已保存到 {output_file_path}")
    print(f"替换文本数量: {num_to_replace}/{num_texts}")