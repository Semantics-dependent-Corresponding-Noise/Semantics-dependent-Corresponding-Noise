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
file_path = '/path/dataset/Entity_Referential_Error_noise_MSCOCO/annotations/scan_split/test_caps.txt'
with open(file_path, 'r', encoding='utf-8') as f:
    raw_texts = f.readlines()


raw_texts = [text.strip() for text in raw_texts]

text_data = pd.DataFrame({'raw': raw_texts})
original_texts = raw_texts.copy()
replace_ratio = 1.0
num_texts = len(raw_texts)
num_to_replace = int(num_texts * replace_ratio)  # Number of texts to replace
indices_to_replace = list(range(num_texts))  

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


batch_size = 50
max_retries = 3
modified_count = 0
target_modified = num_to_replace
used_indices = set()

i = 0
output_test_file_path = '/path/dataset/Entity_Referential_Error_noise_MSCOCO/annotations/scan_split/test_caps_entity.txt'

pbar = tqdm(total=target_modified, desc="Generating noisy captions", unit="caption")
while modified_count < target_modified:

    remaining_indices = list(set(range(num_texts)) - used_indices)
    if not remaining_indices:
        print("No more texts available for modification.")
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
    i += batch_size
    time.sleep(delay_between_requests)
pbar.close()

print(f"Actual number of noisy texts generated: {modified_count}")
with open(output_test_file_path, 'w', encoding='utf-8') as f:
    for text in raw_texts:
        cleaned_text = text.lstrip().split('. ', 1)
        if len(cleaned_text) == 2 and cleaned_text[0].isdigit():
            text = cleaned_text[1]
        f.write(text + "\n")

print(f"Original and modified texts saved to {output_test_file_path}")
