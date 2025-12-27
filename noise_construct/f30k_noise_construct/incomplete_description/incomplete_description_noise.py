import pandas as pd
import ast
import random
from tqdm import tqdm

# Read CSV data
file_path = '/path/flickr_annotations_30k.csv'
data = pd.read_csv(file_path)
train_data = data[data['split'] == 'train']

raw_texts = []
img_ids = []
for row in train_data.itertuples():
    for text in ast.literal_eval(row.raw):
        raw_texts.append(text)
        img_ids.append(row.img_id)

text_data = pd.DataFrame({'raw': raw_texts, 'img_id': img_ids})


# Noise file path
noise_file_path = '/path/dataset/incomplete_description_noise_f30k/annotations/scan_split/1.0_noise_train_caps.txt'

with open(noise_file_path, 'r', encoding='utf-8') as f:
    noise_texts = f.readlines()
noise_texts = [text.strip() for text in noise_texts]

# Replacement ratio
replace_ratio = 0.2  
num_texts = len(raw_texts)
num_to_replace = int(num_texts * replace_ratio) 

indices_to_replace = random.sample(range(num_texts), num_to_replace)

for idx in tqdm(indices_to_replace, desc="Replacing texts"):
    raw_texts[idx] = noise_texts[idx]

# Output file path
output_file_path = f'/path/dataset/incomplete_description_noise_f30k/annotations/scan_split/{replace_ratio}_noise_train_caps.txt'

with open(output_file_path, 'w', encoding='utf-8') as f:
    for text in raw_texts:
        cleaned_text = text.lstrip().split('. ', 1)
        if len(cleaned_text) == 2 and cleaned_text[0].isdigit():
            text = cleaned_text[1]
        f.write(text + "\n")

print(f"Original and modified texts have been saved to {output_file_path}")
