import pandas as pd
import ast
import random
from tqdm import tqdm

# Load CSV data
file_path = '/path/flickr_annotations_30k.csv'
data = pd.read_csv(file_path)

train_data = data[data['split'] == 'train']

raw_texts = []
img_ids = []
text_indices = []  

current_index = 0
for row in train_data.itertuples():
    for text in ast.literal_eval(row.raw):
        raw_texts.append(text)
        img_ids.append(row.img_id)
        text_indices.append(current_index)
        current_index += 1

# Path to the noise file
noise_file_path = '/path/dataset/core_missing_Error_noise_5error_f30k/annotations/scan_split/1.0_noise_train_caps.txt'

with open(noise_file_path, 'r', encoding='utf-8') as f:
    noise_texts = f.readlines()

noise_texts = [text.strip() for text in noise_texts]

# Replacement ratio
replace_ratio = 0.6  


unique_img_ids = list(set(img_ids))
num_images = len(unique_img_ids)
num_images_to_replace = int(num_images * replace_ratio)  


images_to_replace = random.sample(unique_img_ids, num_images_to_replace)
replacement_count = 0

for img_id in tqdm(images_to_replace, desc="Replacing images"):
    indices = [i for i, id_val in enumerate(img_ids) if id_val == img_id]
    for idx in indices:
        if text_indices[idx] < len(noise_texts):
            raw_texts[idx] = noise_texts[text_indices[idx]]
            replacement_count += 1
        else:
            print(f"Warning: No text found at line {text_indices[idx]} in the noise file")

print(f"Total text descriptions replaced: {replacement_count}")

# Output file path
output_file_path = f'/path/dataset/core_missing_Error_noise5error_f30k/annotations/scan_split/{replace_ratio}_noise_train_caps.txt'

with open(output_file_path, 'w', encoding='utf-8') as f:
    for text in raw_texts:
        cleaned_text = text.lstrip().split('. ', 1)
        if len(cleaned_text) == 2 and cleaned_text[0].isdigit():
            text = cleaned_text[1]
        f.write(text + "\n")

print(f"The replaced text has been saved to {output_file_path}")
