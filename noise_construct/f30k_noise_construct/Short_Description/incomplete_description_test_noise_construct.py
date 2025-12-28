import pandas as pd
import ast
import random
from openai import OpenAI
import time
from tqdm import tqdm

# Set KimiChat API key and API endpoint
client = OpenAI(
    api_key="yours api", 
    base_url="https://api.moonshot.cn/v1"
)

# Read text file
file_path = '/path/dataset/incomplete_description_noise_f30k/annotations/scan_split/test_caps.txt'
with open(file_path, 'r', encoding='utf-8') as f:
    raw_texts = f.readlines()
raw_texts = [text.strip() for text in raw_texts]

text_data = pd.DataFrame({'raw': raw_texts})
original_texts = raw_texts.copy()
replace_ratio = 1.0
num_texts = len(raw_texts)
num_to_replace = int(num_texts * replace_ratio)  
indices_to_replace = list(range(num_texts))      


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


batch_size = 50
max_retries = 3
modified_count = 0
target_modified = num_to_replace
used_indices = set()

i = 0
pbar = tqdm(total=target_modified, desc="Generating noisy captions", unit="caption")
while modified_count < target_modified:
    i += 1
    remaining_indices = list(set(range(num_texts)) - used_indices)
    if not remaining_indices:
        print("No more available texts to modify.")
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
            print(f"Batch {i} failed after {max_retries} retries, skipping.")
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

print(f"Actual number of noisy texts generated: {modified_count}")

output_test_file_path = '/path/dataset/incomplete_description_noise_f30k/annotations/test_caps.txt'

with open(output_test_file_path, 'w', encoding='utf-8') as f:
    for text in raw_texts:
        cleaned_text = text.lstrip().split('. ', 1)
        if len(cleaned_text) == 2 and cleaned_text[0].isdigit():
            text = cleaned_text[1]
        f.write(text + "\n")

print(f"Original and modified texts have been saved to {output_test_file_path}")
