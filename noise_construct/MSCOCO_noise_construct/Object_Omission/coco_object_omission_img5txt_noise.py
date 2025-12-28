import random
from tqdm import tqdm

# Read raw text data
file_path = '/path/dataset/Object_Omission_noise_5error_MSCOCO/annotations/scan_split/0_noise_train_caps.txt'
with open(file_path, 'r', encoding='utf-8') as f:
    raw_texts = f.readlines()
raw_texts = [text.strip() for text in raw_texts]

# Path to noise file
noise_file_path = '/path/dataset/Object_Omission_noise_5error_MSCOCO/annotations/scan_split/1.0_noise_train_caps.txt'

with open(noise_file_path, 'r', encoding='utf-8') as f:
    noise_texts = f.readlines()
noise_texts = [text.strip() for text in noise_texts]


# Replacement ratio
replace_ratio = 0.6

descriptions_per_image = 5
total_images = len(raw_texts) // descriptions_per_image


# Calculate the number of images requiring replacement
num_images_to_replace = int(total_images * replace_ratio)
images_to_replace = random.sample(range(total_images), num_images_to_replace)
replacement_count = 0

for img_idx in tqdm(images_to_replace, desc="Replacing images"):
    start_idx = img_idx * descriptions_per_image
    end_idx = start_idx + descriptions_per_image
    
    if end_idx <= len(raw_texts):
        for i in range(start_idx, end_idx):
            if i < len(noise_texts):
                raw_texts[i] = noise_texts[i]
                replacement_count += 1
            else:
                print(f"Warning: Text for line {i} is missing from the noise file.")

print(f"A total of {replacement_count} text descriptions have been replaced.")
print(f"Involving {num_images_to_replace} images")

# Output file path
output_file_path = f'/path/dataset/Object_Omission_noise_5error_MSCOCO/annotations/scan_split/{replace_ratio}_noise_train_caps.txt'
with open(output_file_path, 'w', encoding='utf-8') as f:
    for text in raw_texts:
        cleaned_text = text.lstrip().split('. ', 1)
        if len(cleaned_text) == 2 and cleaned_text[0].isdigit():
            text = cleaned_text[1]
        f.write(text + "\n")

print(f"The replaced text has been saved to {output_file_path}")
