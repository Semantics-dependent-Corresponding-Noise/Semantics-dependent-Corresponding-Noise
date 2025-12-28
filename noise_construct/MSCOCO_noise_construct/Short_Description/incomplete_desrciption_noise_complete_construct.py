import random
from openai import OpenAI
import time
from tqdm import tqdm
import os
import logging
import sys
from datetime import datetime

def setup_logging(log_dir="logs"):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"noise_generation_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    return log_file

# Set KimiChat API key and API endpoint
client = OpenAI(
    api_key="yours api",
    base_url="https://api.moonshot.cn/v1"
)

class SimpleProgress:
    def __init__(self, total, desc="Progress"):
        self.total = total
        self.desc = desc
        self.current = 0
        self.last_log_percent = -1
        self.start_time = time.time()
        
    def update(self, n=1):
        self.current += n
        current_percent = int(self.current / self.total * 100)
        
        if current_percent != self.last_log_percent and (current_percent % 5 == 0 or current_percent == 100):
            elapsed = time.time() - self.start_time
            if self.current > 0:
                eta = elapsed * (self.total - self.current) / self.current
                eta_str = f"ETA: {eta:.1f}s"
            else:
                eta_str = "ETA: calculating..."
                
            logging.info(f"{self.desc}: {self.current}/{self.total} ({current_percent}%) {eta_str}")
            self.last_log_percent = current_percent
    
    def close(self):
        elapsed = time.time() - self.start_time
        logging.info(f"{self.desc}: Completed {self.current}/{self.total} (100%) Time: {elapsed:.2f}s")

def compare_and_modify_files_fast(file1_path, file2_path, file3_path, replace_ratio=1.0):
    
    logging.info("Starting file comparison and modification (optimized version)")
    logging.info(f"File 1 (Reference): {file1_path}")
    logging.info(f"File 2 (Input): {file2_path}")
    logging.info(f"File 3 (Output): {file3_path}")
    
    if not os.path.exists(file1_path):
        logging.error(f"File 1 does not exist: {file1_path}")
        return
    if not os.path.exists(file2_path):
        logging.error(f"File 2 does not exist: {file2_path}")
        return
    
    logging.info("Reading File 1...")
    try:
        with open(file1_path, 'r', encoding='utf-8') as f:
            file1_lines = [line.strip() for line in f if line.strip()]
        file1_set = set(file1_lines)  
        logging.info(f"Successfully read File 1, line count: {len(file1_lines)}")
    except Exception as e:
        logging.error(f"Failed to read File 1: {e}")
        return
    
    logging.info("Reading File 2...")
    try:
        with open(file2_path, 'r', encoding='utf-8') as f:
            file2_lines = [line.strip() for line in f if line.strip()]
        logging.info(f"Successfully read File 2, line count: {len(file2_lines)}")
    except Exception as e:
        logging.error(f"Failed to read File 2: {e}")
        return
    

    logging.info("Comparing file contents quickly...")
    matching_indices = []
    
    is_tty = sys.stdout.isatty()
    if is_tty:
        pbar = tqdm(total=len(file2_lines), desc="Comparing files", unit="line")
    else:
        pbar = SimpleProgress(len(file2_lines), desc="Comparing files")
    
    for i, line2 in enumerate(file2_lines):
        if line2 in file1_set:  
            matching_indices.append(i)
        pbar.update(1)
    
    pbar.close()
    
    logging.info(f"Found {len(matching_indices)} identical lines")
    
    if not matching_indices:
        logging.warning("No identical lines found, directly copying File 2 to File 3")
        try:
            with open(file3_path, 'w', encoding='utf-8') as f:
                for line in file2_lines:
                    f.write(line + "\n")
            logging.info(f"File saved: {file3_path}")
        except Exception as e:
            logging.error(f"Failed to save file: {e}")
        return
    

    num_to_modify = int(len(matching_indices) * replace_ratio)
    indices_to_modify = random.sample(matching_indices, num_to_modify)
    logging.info(f"Planning to modify {num_to_modify} identical lines")
    
    if num_to_modify == 0:
        logging.info("No modification needed, saving file directly")
        try:
            with open(file3_path, 'w', encoding='utf-8') as f:
                for line in file2_lines:
                    f.write(line + "\n")
            logging.info(f"File saved: {file3_path}")
        except Exception as e:
            logging.error(f"Failed to save file: {e}")
        return
    
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
            logging.debug(f"Sending API request, batch size: {len(text_list)}")
            completion = client.chat.completions.create(
                model="moonshot-v1-8k",
                messages=[
                    {"role": "system", "content": prompt1},
                    {"role": "user", "content": prompt2}
                ],
                temperature=0.3,
                timeout=60
            )
            processed_results = []
            for line in completion.choices[0].message.content.strip().split('\n'):
                line = line.strip()
                if line:
                    if '. ' in line and line.split('. ')[0].isdigit():
                        line = line.split('. ', 1)[1]
                    processed_results.append(line)
            logging.debug(f"API response processed, returned {len(processed_results)} results")
            return processed_results
        except Exception as e:
            logging.error(f"API call error: {e}")
            return text_list

    requests_per_minute = 50
    delay_between_requests = 60 / requests_per_minute


    batch_size = 20  
    max_retries = 3
    modified_count = 0


    modified_file2_lines = file2_lines.copy()


    if is_tty:
        main_pbar = tqdm(total=num_to_modify,  # Use correct variable name num_to_modify
                        desc="Modifying identical lines", 
                        unit="text",
                        ncols=100)
    else:
        main_pbar = SimpleProgress(num_to_modify, desc="Modifying identical lines")

    batch_count = 0
    total_batches = (num_to_modify + batch_size - 1) // batch_size
    logging.info(f"Batch Info - Total batches: {total_batches}, Batch size: {batch_size}")
    

    remaining_indices = indices_to_modify.copy()
    
    while modified_count < num_to_modify and remaining_indices:

        current_batch_size = min(batch_size, len(remaining_indices))
        current_batch_indices = remaining_indices[:current_batch_size]
        remaining_indices = remaining_indices[current_batch_size:]
        
        
        batch_count += 1
        logging.info(f"Processing batch {batch_count}/{total_batches}")
        
        retry_count = 0
        failed_indices = current_batch_indices.copy()
        
        while retry_count < max_retries and failed_indices:
            try:
                current_failed_texts = [modified_file2_lines[idx] for idx in failed_indices]
                
                logging.info(f"Retry {retry_count + 1}: Sending {len(failed_indices)} texts to API")
                batch_modified = generate_noisy_text_batch(current_failed_texts)
                
                if len(batch_modified) == len(failed_indices):
                    new_failed_indices = []
                    success_in_batch = 0
                    
                    for j, idx in enumerate(failed_indices):
                        original_text = modified_file2_lines[idx]
                        modified_text = batch_modified[j]
                        
                        if modified_text and modified_text != original_text:
                            modified_file2_lines[idx] = modified_text
                            modified_count += 1
                            success_in_batch += 1
                            main_pbar.update(1)
                        else:
                            new_failed_indices.append(idx)
                    
                    failed_indices = new_failed_indices
                    logging.info(f"Batch results - Success: {success_in_batch}, Failed: {len(failed_indices)}")
                    
                    if not failed_indices:
                        logging.info("All texts in current batch processed successfully")
                        break  
                        
                else:
                    logging.warning(f"API return count mismatch: Expected {len(failed_indices)}, Got {len(batch_modified)}")
                    failed_indices = []  
                    
            except Exception as e:
                logging.error(f"API call exception: {e}")
            
            retry_count += 1
            if retry_count < max_retries and failed_indices:
                logging.info(f"Preparing retry {retry_count}, remaining {len(failed_indices)} texts")
                time.sleep(5)
        
        success_count = len(current_batch_indices) - len(failed_indices)
        if success_count > 0:
            logging.info(f"Batch complete - Successfully modified: {success_count} texts")
        if failed_indices:
            logging.warning(f"Batch complete - Kept original texts: {len(failed_indices)} texts")
        

        if remaining_indices:
            time.sleep(delay_between_requests)
    
    main_pbar.close()

    logging.info(f"Modification complete - Actually modified: {modified_count}/{num_to_modify}")

    try:
        with open(file3_path, 'w', encoding='utf-8') as f:
            for text in modified_file2_lines:
                f.write(text + "\n")
        logging.info(f"File saved successfully: {file3_path}")
    except Exception as e:
        logging.error(f"File save failed: {e}")
        return

    logging.info("File comparison and modification complete")
    logging.info("=" * 70)

if __name__ == "__main__":

    log_file = setup_logging()
    
    # Configuration parameters
    file1_path = "/path/dataset/incomplete_description_noise_MSCOCO/annotations/scan_split/0_noise_train_caps.txt"
    file2_path = "/path/noise_construct/MSCOCO_noise_cinstruct/incomplete_description/1.0_noise_train_caps_preliminary.txt"
    file3_path = "/path/dataset/incomplete_description_noise_MSCOCO/annotations/scan_split/1.0_noise_train_caps.txt"
    replace_ratio = 1.0
    
    logging.info("=" * 80)
    logging.info("Starting file comparison and modification (optimized version)")
    logging.info(f"Reference File: {file1_path}")
    logging.info(f"Input File: {file2_path}")
    logging.info(f"Output File: {file3_path}")
    logging.info(f"Replacement Ratio: {replace_ratio}")
    logging.info(f"Log File: {log_file}")
    logging.info("=" * 80)
    
    start_time = time.time()
    
    try:
        compare_and_modify_files_fast(file1_path, file2_path, file3_path, replace_ratio)
        
        end_time = time.time()
        total_time = end_time - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        logging.info(f"Program execution complete - Total time: {hours:02d}:{minutes:02d}:{seconds:02d}")
        logging.info("File comparison and modification complete!")
        
    except KeyboardInterrupt:
        logging.warning("Program interrupted by user")
    except Exception as e:
        logging.error(f"Program execution error: {e}", exc_info=True)
        raise
