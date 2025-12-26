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
num_to_replace = num_texts  # 修改为全部文本
indices_to_replace = list(range(num_texts))  # 按顺序处理全部文本
# num_to_replace = 100  
# indices_to_replace = random.sample(range(num_texts), num_to_replace)  # 随机选择1000个文本
# 批量处理函数
def generate_noisy_text_batch(text_list):
    prompt1 = """You are a professional Sentence revision Assistant., and your only task is to condense sentences without altering their essential meaning. Please strictly follow the following rules and output format:
Core Rules for Condense sentence:
1.Extraction of the sentence subject: First, accurately identify every subject component in the input sentence (for example, object category, color, scene, action，numerical expressions).
2.Partial removal or simplification of sentence components: Remove some sentence components to make the sentence more concise. Retain some descriptive words of the components, but ensure the sentence no longer fully describes the original scene. Avoid altering the verb structure (e.g., do not remove or change the form of the verb).
3.Ensure the Modified Sentence Omits at Least One Key Action or Detail: The modified sentence should omit at least one key action or detail, making it less descriptive than the original. The action or detail omitted should result in a change in the meaning or completeness of the sentence.
4.Avoid simply removing adjectives, adverbs, or other modifying elements: the revised sentence should not be entirely identical to the original in meaning. For instance, When multiple subjects appear in parallel, such as concurrent actions, parallel subjects or objects, one or more may be omitted, but at least one must be retained, resulting in the sentence describing a default state.
5.Where none of the above rules can be applied to modify a sentence, simply return the subject of the original sentence.
- Input Sentence: A man in a pink shirt climbs a rock face.
- Output Sentence: A man in a pink shirt.
- Input Sentence: A boys jumps into the water upside down.
- Output Sentence: A boys jumps into the water.
- Input Sentence: This is a young boy playing with a dollhouse.
- Output Sentence: A young boy.
- Input Sentence: A man wearing a cap and glasses is fixing the seat of a bicycle.
- Output Sentence: A man wearing a cap is fixing the seat of a bicycle.
- Input Sentence:A young boy is frantically staring and shaking his hands.
- Output Sentence: A young boy is frantically staring.
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
batch_size = 20
max_retries = 3
modified_count = 0
target_modified = num_to_replace
used_indices = set()
failed_indices = set()  # 用于记录失败的索引

i = 0
pbar = tqdm(total=target_modified, desc="Generating noisy captions", unit="caption")
while modified_count < target_modified:
    # 采样未被修改过的索引，优先处理失败的索引
    remaining_indices = list(set(range(num_texts)) - used_indices)
    if failed_indices:  # 如果有失败的索引，优先处理这些索引
        remaining_indices = list(failed_indices)
        failed_indices = set()  # 清空失败的索引，准备重新尝试
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
            print(f"Batch {i//batch_size+1} failed after {max_retries} retries, retrying later.")
            failed_indices.update(batch_indices)  # 将失败的索引记录下来
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
    i += 1
pbar.close()

print(f"实际生成噪声文本数量: {modified_count}")

output_test_file_path = f'/data/xjd/study/NPC/dataset/incomplete_description/annotations/scan_split/{replace_ratio}_noise_train_caps2.txt'
# output_test_file_path = f'/data/xjd/study/NPC/dataset/incomplete_description/annotations/scan_split/1000_noise_train_caps_test.txt'

with open(output_test_file_path, 'w', encoding='utf-8') as f:
    for text in raw_texts:
        # 去除开头的编号和点（如 "1. "）
        cleaned_text = text.lstrip().split('. ', 1)
        if len(cleaned_text) == 2 and cleaned_text[0].isdigit():
            text = cleaned_text[1]
        f.write(text + "\n")
# max_retries = 3

# def get_modified_text(original_text):
#     for _ in range(max_retries):
#         modified = generate_noisy_text_batch([original_text])[0]
#         # 去除编号
#         cleaned = modified.lstrip().split('. ', 1)
#         if len(cleaned) == 2 and cleaned[0].isdigit():
#             modified = cleaned[1]
#         if modified.strip() != original_text.strip():
#             return modified
#     return modified  # 如果多次都没变，返回最后一次结果
# with open(output_test_file_path, 'w', encoding='utf-8') as f:
#     for idx in indices_to_replace:
#         original_text = original_texts[idx]
#         modified_text = get_modified_text(original_text)
#         f.write(modified_text + "\n")

# with open(output_test_file_path, 'w', encoding='utf-8') as f:
#     # 处理一个文本并写入原始文本和修改后的文本
#     for idx in used_indices:
#         original_text = original_texts[idx]  # 从原始文本副本中获取原始文本
#         modified_text = raw_texts[idx]  # 从修改后的文本列表中获取修改后的文本
#         # 写入原始文本
#         f.write(f"Original Text: {original_text}\n")
#         # 写入修改后的文本
#         f.write(f"Modified Text: {modified_text}\n")
#         # 写入一个空行以分隔不同的文本对
#         f.write("\n")

print(f"原始文本和修改后的文本已保存到 {output_test_file_path}")
