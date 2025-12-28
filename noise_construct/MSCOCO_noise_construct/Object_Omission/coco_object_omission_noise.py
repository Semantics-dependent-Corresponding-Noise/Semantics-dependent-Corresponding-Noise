import random
from tqdm import tqdm

# Read raw text data
file_path = '/path/dataset/Object_Omission_noise_MSCOCO/annotations/scan_split/0_noise_train_caps.txt'
with open(file_path, 'r', encoding='utf-8') as f:
    raw_texts = f.readlines()
raw_texts = [text.strip() for text in raw_texts]

# Path to noise file
noise_file_path = '/path/dataset/Object_Omission_noise_MSCOCO/annotations/scan_split/1.0_noise_train_caps.txt'
with open(noise_file_path, 'r', encoding='utf-8') as f:
    noise_texts = f.readlines()
noise_texts = [text.strip() for text in noise_texts]

# Multiple replacement ratios
replace_ratios = [0.2,0.4,0.6]

for replace_ratio in replace_ratios:
    modified_texts = raw_texts.copy()
    num_texts = len(modified_texts)
    num_to_replace = int(num_texts * replace_ratio)
    indices_to_replace = random.sample(range(num_texts), num_to_replace)
    for idx in tqdm(indices_to_replace, desc=f"Replacing texts ({replace_ratio})"):
        modified_texts[idx] = noise_texts[idx]
    
    # Output file path
    output_file_path = f'/path/dataset/Object_Omission_noise_MSCOCO/annotations/scan_split/{replace_ratio}_noise_train_caps.txt'

    with open(output_file_path, 'w', encoding='utf-8') as f:
        for text in modified_texts:
            cleaned_text = text.lstrip().split('. ', 1)
            if len(cleaned_text) == 2 and cleaned_text[0].isdigit():
                text = cleaned_text[1]
            f.write(text + "\n")
    
    print(f"The text with a replacement ratio of {replace_ratio} has been saved to {output_file_path}.")
    print(f"Number of replacement texts: {num_to_replace}/{num_texts}")
