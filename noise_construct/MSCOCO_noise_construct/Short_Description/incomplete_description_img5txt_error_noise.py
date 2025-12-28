import random
from tqdm import tqdm

# Read original text data
file_path = '/path/dataset/incomplete_description_noise_5error_MSCOCO/annotations/scan_split/0_noise_train_caps.txt'
with open(file_path, 'r', encoding='utf-8') as f:
    raw_texts = f.readlines()
raw_texts = [text.strip() for text in raw_texts]

# Noise file path
noise_file_path = '/path/dataset/incomplete_description_noise_5error_MSCOCO/annotations/scan_split/1.0_noise_train_caps.txt'

with open(noise_file_path, 'r', encoding='utf-8') as f:
    noise_texts = f.readlines()

noise_texts = [text.strip() for text in noise_texts]


if len(noise_texts) < len(raw_texts):
    print(f"Warning: Number of noise texts ({len(noise_texts)}) is less than raw texts ({len(raw_texts)})")
    noise_texts = noise_texts * (len(raw_texts) // len(noise_texts) + 1)

# Replacement ratio
replace_ratio = 0.2

descriptions_per_image = 5
total_images = len(raw_texts) // descriptions_per_image

print(f"Total number of images: {total_images}")
print(f"Total number of texts: {len(raw_texts)}")
print(f"Number of captions per image: {descriptions_per_image}")

num_images_to_replace = int(total_images * replace_ratio)

images_to_replace = random.sample(range(total_images), num_images_to_replace)

print(f"Number of images to replace: {num_images_to_replace}")


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
                print(f"Warning: No text found in noise file at line {i}")

print(f"Total text descriptions replaced: {replacement_count}")
print(f"Involving {num_images_to_replace} images")

# Output file path
output_file_path = f'/path/dataset/incomplete_description_noise_5error_MSCOCO/annotations/scan_split/{replace_ratio}_noise_train_caps.txt'

with open(output_file_path, 'w', encoding='utf-8') as f:
    for text in raw_texts:
        cleaned_text = text.lstrip().split('. ', 1)
        if len(cleaned_text) == 2 and cleaned_text[0].isdigit():
            text = cleaned_text[1]
        f.write(text + "\n")

print(f"Replaced text saved to {output_file_path}")
