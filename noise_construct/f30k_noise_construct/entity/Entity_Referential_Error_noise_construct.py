import pandas as pd
import ast
import random
from openai import OpenAI
import time
from tqdm import tqdm

# 设置 KimiChat API 密钥和 API 端点
client = OpenAI(
    api_key="yours api",
    base_url="https://api.moonshot.cn/v1"
)

# 读取Excel数据
file_path = '/data/xjd/study/NPC/flickr_annotations_30k.csv'
data = pd.read_csv(file_path)

# 提取训练集文本
train_data = data[data['split'] == 'train']

# 将第一列的文本展开为单独的行，并保留img_id
raw_texts = []
img_ids = []
for row in train_data.itertuples():
    for text in ast.literal_eval(row.raw):
        raw_texts.append(text)
        img_ids.append(row.img_id)

# 创建一个DataFrame来存储文本和对应的图片编号
text_data = pd.DataFrame({'raw': raw_texts, 'img_id': img_ids})
original_texts = raw_texts.copy()
replace_ratio = 1.0
num_texts = len(raw_texts)
# num_to_replace = 12
num_to_replace = int(num_texts * replace_ratio)  # 需要替换的文本数量
indices_to_replace = list(range(num_texts))      # 按顺序处理全部文本

# 批量处理函数
def generate_noisy_text_batch(text_list):
    prompt1 = """You are a professional Sentence revision Assistant., and your only task is to replace the entities referred to in the sentence without altering their essential meaning. Please strictly follow the following rules and output format:
Core Rules for Replace the entities referred:
1.Extraction of the sentence subject: First, accurately identify every subject component in the input sentence (for example,persons, objects, locations).
2.Replace Entities with Synonyms: Replace each identified entity with a different but similar entity within the same category. Ensure the replacement is reasonable but results in a change in meaning.(for example, replace a cat with a dog, a boy with a girl, ).  
3.Maintain Logical Coherence: The modified sentence should still make logical sense, but the meaning should be distinct from the original.
4.Flexible Replacement: Allow for broader replacements of entities to accommodate different contexts and requirements.
Example of Entities Referred Replacement:
- Input Sentence: People buying food from a street vendor.
- Output Sentence: People buying food from a restaurant.
- Input Sentence: A boys jumps into the water upside down.
- Output Sentence: A girls jumps into the water upside down.
- Input Sentence: A brown dog is licking its nose.
- Output Sentence: A brown cat is licking its nose.
- Input Sentence: A professor in front of his class giving a lecture.
- Output Sentence: A student in front of her class giving a presentation.
- Input Sentence: A woman in glasses plays guitar.
- Output Sentence: A man in glasses plays piano.
Strict Output Format:
Only output the modified sentence directly. Do NOT add any extra content (such as explanations, notes, or greetings)."""
    prompt2 = (
        "Please process the following sentences in batches according to the rules, outputting one modified sentence per line:\n"
        + "\n".join([f"{i+1}. {text}" for i, text in enumerate(text_list)])
    )
    try:
        completion = client.chat.completions.create(
            model="moonshot-v1-8k",
            messages=[
                {"role": "system", "content": prompt1},
                {"role": "user", "content": prompt2}
            ],
            temperature=0.3,
        )
        return [line.strip() for line in completion.choices[0].message.content.strip().split('\n') if line.strip()]
    except Exception as e:
        print(f"Error calling the API: {e}")
        return text_list

requests_per_minute = 5000
delay_between_requests = 60 / requests_per_minute

# 批次处理
batch_size = 50
max_retries = 3
modified_count = 0
target_modified = num_to_replace
used_indices = set()

i = 0
pbar = tqdm(total=target_modified, desc="Generating noisy captions", unit="caption")
while modified_count < target_modified:
    # 采样未被修改过的索引
    remaining_indices = list(set(range(num_texts)) - used_indices)
    if not remaining_indices:
        print("没有更多可用的文本进行修改。")
        break
    batch_indices = random.sample(remaining_indices, min(batch_size, len(remaining_indices)))
    batch_texts = [raw_texts[idx] for idx in batch_indices]
    retry_count = 0
    while True:
        batch_modified = generate_noisy_text_batch(batch_texts)
        if len(batch_modified) == len(batch_indices):
            break
        retry_count += 1
        if retry_count >= max_retries:
            print(f"Batch {i//batch_size+1} failed after {max_retries} retries, skipping.")
            batch_modified = []
            break
        time.sleep(2)
    for j, idx in enumerate(batch_indices):
        if j < len(batch_modified) and batch_modified[j] != raw_texts[idx]:
            raw_texts[idx] = batch_modified[j]
            used_indices.add(idx)
            modified_count += 1
            pbar.update(1)
            if modified_count >= target_modified:
                break
    time.sleep(delay_between_requests)
pbar.close()

print(f"实际生成噪声文本数量: {modified_count}")

output_test_file_path = f'/data/xjd/study/NPC/dataset/Entity_Referential_Error/annotations/scan_split/{replace_ratio}_noise_train_caps.txt'

with open(output_test_file_path, 'w', encoding='utf-8') as f:
    for text in raw_texts:
        # 去除开头的编号和点（如 "1. "）
        cleaned_text = text.lstrip().split('. ', 1)
        if len(cleaned_text) == 2 and cleaned_text[0].isdigit():
            text = cleaned_text[1]
        f.write(text + "\n")

print(f"原始文本和修改后的文本已保存到 {output_test_file_path}")
