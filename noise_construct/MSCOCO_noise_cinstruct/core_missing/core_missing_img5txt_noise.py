import random
from tqdm import tqdm

# 读取原始文本数据
file_path = '/home/zbm/xjd/NPC-master/dataset/core_missing_Error_noise_5eeror_coco/annotations/scan_split/0_noise_train_caps.txt'

# 读取原始文本文件
with open(file_path, 'r', encoding='utf-8') as f:
    raw_texts = f.readlines()

# 去除文本中的换行符
raw_texts = [text.strip() for text in raw_texts]

# 噪声文件路径
noise_file_path = '/home/zbm/xjd/NPC-master/dataset/core_missing_Error_noise_5eeror_coco/annotations/scan_split/1.0_noise_train_caps.txt'

# 读取噪声文件
with open(noise_file_path, 'r', encoding='utf-8') as f:
    noise_texts = f.readlines()

# 去除噪声文本中的换行符
noise_texts = [text.strip() for text in noise_texts]

# 确保噪声文本数量足够
if len(noise_texts) < len(raw_texts):
    print(f"警告: 噪声文本数量({len(noise_texts)})少于原始文本数量({len(raw_texts)})")
    noise_texts = noise_texts * (len(raw_texts) // len(noise_texts) + 1)

# 替换比例
replace_ratio = 0.6

# 假设每个图像有5条描述文本，且连续存储
descriptions_per_image = 5
total_images = len(raw_texts) // descriptions_per_image

print(f"总图像数量: {total_images}")
print(f"总文本数量: {len(raw_texts)}")
print(f"每个图像的描述文本数量: {descriptions_per_image}")

# 计算需要替换的图像数量
num_images_to_replace = int(total_images * replace_ratio)

# 随机选择需要替换的图像索引
images_to_replace = random.sample(range(total_images), num_images_to_replace)

print(f"需要替换的图像数量: {num_images_to_replace}")

# 替换文本：将选中图像的所有描述文本替换为噪声文件中对应位置的文本
replacement_count = 0

for img_idx in tqdm(images_to_replace, desc="Replacing images"):
    # 计算该图像对应的文本起始索引
    start_idx = img_idx * descriptions_per_image
    end_idx = start_idx + descriptions_per_image
    
    # 确保索引不越界
    if end_idx <= len(raw_texts):
        # 替换该图像的所有描述文本
        for i in range(start_idx, end_idx):
            if i < len(noise_texts):
                raw_texts[i] = noise_texts[i]
                replacement_count += 1
            else:
                print(f"警告: 噪声文件中没有第 {i} 行的文本")

print(f"总共替换了 {replacement_count} 条文本描述")
print(f"涉及 {num_images_to_replace} 个图像")

# 输出文件路径
output_file_path = f'/home/zbm/xjd/NPC-master/dataset/core_missing_Error_noise_5eeror_coco/annotations/scan_split/{replace_ratio}_noise_train_caps.txt'

# 保存替换后的文本到文件
with open(output_file_path, 'w', encoding='utf-8') as f:
    for text in raw_texts:
        # 去除开头的编号和点（如 "1. "）
        cleaned_text = text.lstrip().split('. ', 1)
        if len(cleaned_text) == 2 and cleaned_text[0].isdigit():
            text = cleaned_text[1]
        f.write(text + "\n")

print(f"替换后的文本已保存到 {output_file_path}")