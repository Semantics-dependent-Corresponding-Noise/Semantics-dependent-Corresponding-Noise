import pandas as pd
import ast
import random
from openai import OpenAI
import time
from tqdm import tqdm
import os
import glob
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

# Set API key and API endpoint
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

def process_single_file(input_file_path, output_file_path, replace_ratio=1.0):
    """
    Process a single text file, generate noisy text, and save it.
    
    Args:
        input_file_path: Path to the input file
        output_file_path: Path to the output file
        replace_ratio: Ratio of text to replace
    """
    
    logging.info(f"Started processing file: {input_file_path}")
    
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            raw_texts = f.readlines()
    except Exception as e:
        logging.error(f"Failed to read file: {e}")
        return
    raw_texts = [text.strip() for text in raw_texts if text.strip()]
    
    if not raw_texts:
        logging.warning(f"File is empty, skipping: {input_file_path}")
        return

    logging.info(f"File Info - Filename: {os.path.basename(input_file_path)}, Text Count: {len(raw_texts)}")

    original_texts = raw_texts.copy()
    
    num_texts = len(raw_texts)
    num_to_replace = int(num_texts * replace_ratio)  

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

    batch_size = 50
    max_retries = 5
    modified_count = 0
    target_modified = num_to_replace
    used_indices = set()


    is_tty = sys.stdout.isatty()
    
    if is_tty:
        logging.info("Using interactive progress bar")
        main_pbar = tqdm(total=target_modified, 
                        desc=f"Processing {os.path.basename(input_file_path)}", 
                        unit="text",
                        ncols=100,
                        bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}')
    else:
        logging.info(f"Starting noise generation, target count: {target_modified}")
        main_pbar = SimpleProgress(target_modified, desc=f"Processing {os.path.basename(input_file_path)}")

    batch_count = 0
    total_batches = (target_modified + batch_size - 1) // batch_size
    logging.info(f"Batch Info - Total batches: {total_batches}, Batch size: {batch_size}")
    
    while modified_count < target_modified:
        remaining_indices = list(set(range(num_texts)) - used_indices)
        if not remaining_indices:
            logging.warning("No more texts available for modification")
            break
            
        batch_indices = random.sample(remaining_indices, min(batch_size, len(remaining_indices)))
        logging.debug(f"Sampled batch indices: {len(batch_indices)} texts")
        
        retry_count = 0
        failed_indices = batch_indices.copy()  # Initially consider all as failed (pending)
        
        batch_count += 1
        logging.info(f"Processing batch {batch_count}/{total_batches}")
        
        while retry_count < max_retries and failed_indices:
            try:
                current_batch_texts = [raw_texts[idx] for idx in failed_indices]
                
                logging.info(f"Retry {retry_count + 1}: Sending {len(failed_indices)} texts to API")
                batch_modified = generate_noisy_text_batch(current_batch_texts)
                
                if len(batch_modified) == len(failed_indices):
                    new_failed_indices = []
                    success_in_batch = 0
                    
                    for j, idx in enumerate(failed_indices):
                        original_text = raw_texts[idx]
                        modified_text = batch_modified[j]
                        
                        if modified_text and modified_text != original_text:
                            raw_texts[idx] = modified_text
                            used_indices.add(idx)
                            modified_count += 1
                            success_in_batch += 1
                            

                            if is_tty:
                                main_pbar.update(1)
                                main_pbar.set_description(f"Processing {os.path.basename(input_file_path)} ({modified_count}/{target_modified})")
                            else:
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
                    
            except Exception as e:
                logging.error(f"API call exception: {e}")
            
            retry_count += 1
            if retry_count < max_retries and failed_indices:
                logging.info(f"Preparing retry {retry_count}, remaining {len(failed_indices)} texts")
                time.sleep(5)
        

        success_count = len(batch_indices) - len(failed_indices)
        if success_count > 0:
            logging.info(f"Batch complete - Successfully modified: {success_count} texts")
        if failed_indices:
            logging.warning(f"Batch complete - Kept original texts: {len(failed_indices)} texts")
            for idx in failed_indices:
                used_indices.add(idx)
        
        time.sleep(delay_between_requests)
    

    main_pbar.close()

    logging.info(f"File processing complete - Actually generated noise: {modified_count}/{num_texts}")

    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for text in raw_texts:
                f.write(text + "\n")
        logging.info(f"File saved successfully: {output_file_path}")
    except Exception as e:
        logging.error(f"File save failed: {e}")
        return

    logging.info(f"Single file processing complete: {os.path.basename(input_file_path)}")
    logging.info("=" * 70)

def process_split_files_sequential(input_dir, output_dir, start_num=1, end_num=None, replace_ratio=1.0):
    """
    Process split files sequentially (split_file_001.txt, split_file_002.txt, ...)
    
    Args:
        input_dir: Input file directory
        output_dir: Output file directory
        start_num: Start file number
        end_num: End file number
        replace_ratio: Replacement ratio
    """
    
    logging.info(f"Starting batch processing - Input dir: {input_dir}, Output dir: {output_dir}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Creating output directory: {output_dir}")
    
    if end_num is None:
        # Automatically detect file count
        existing_files = glob.glob(os.path.join(input_dir, "split_file_*.txt"))
        if existing_files:
            file_numbers = [int(f.split('_')[-1].split('.')[0]) for f in existing_files]
            end_num = max(file_numbers)
            logging.info(f"Automatically detected file range: 001 to {end_num:03d}")
        else:
            logging.error("No split_file_*.txt files found")
            return
    
    logging.info(f"Processing file range: {start_num:03d} to {end_num:03d}, Total {end_num - start_num + 1} files")
    
    is_tty = sys.stdout.isatty()
    
    if is_tty:
        total_files = end_num - start_num + 1
        overall_pbar = tqdm(total=total_files, 
                           desc="Overall Progress", 
                           unit="file",
                           position=0,
                           ncols=100)
    else:

        total_files = end_num - start_num + 1
        overall_pbar = SimpleProgress(total_files, desc="Overall File Progress")
    

    processed_count = 0
    for file_num in range(start_num, end_num + 1):
        input_filename = f"split_file_{file_num:03d}.txt"
        input_file_path = os.path.join(input_dir, input_filename)
        
        if not os.path.exists(input_file_path):
            logging.warning(f"File does not exist, skipping: {input_file_path}")
            overall_pbar.update(1)
            continue
            
        output_filename = f"noisy_split_file_{file_num:03d}.txt"
        output_file_path = os.path.join(output_dir, output_filename)
        
        if os.path.exists(output_file_path):
            overall_pbar.update(1)
            continue
        
        logging.info(f"Started processing file: {input_filename}")
        file_start_time = time.time()
        
        process_single_file(input_file_path, output_file_path, replace_ratio)
        
        processed_count += 1
        overall_pbar.update(1)
        
        file_elapsed_time = time.time() - file_start_time
        logging.info(f"File processing complete: {input_filename}, Time: {file_elapsed_time:.2f}s")
        
        time.sleep(1)

    overall_pbar.close()
    
    logging.info(f"Batch processing complete - Processed {processed_count} files")


if __name__ == "__main__":

    log_file = setup_logging()
    
    # Configuration Parameters
    input_directory = "/path/noise_construct/MSCOCO_noise_cinstruct/incomplete_description/original"
    output_directory = "/path/noise_construct/MSCOCO_noise_cinstruct/incomplete_description/noise"
    replace_ratio = 1.0
    
    logging.info("=" * 80)
    logging.info("Starting batch processing of split text files")
    logging.info(f"Input directory: {input_directory}")
    logging.info(f"Output directory: {output_directory}")
    logging.info(f"Replacement ratio: {replace_ratio}")
    logging.info(f"Log file: {log_file}")
    logging.info("=" * 80)
    
    start_time = time.time()
    
    try:
        process_split_files_sequential(input_directory, output_directory, 
                                     start_num=1, end_num=None,
                                     replace_ratio=replace_ratio)
        
        end_time = time.time()
        total_time = end_time - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        logging.info(f"Program execution complete - Total time: {hours:02d}:{minutes:02d}:{seconds:02d}")
        logging.info("All files processed!")
        
    except KeyboardInterrupt:
        logging.warning("Program interrupted by user")
    except Exception as e:
        logging.error(f"Program execution error: {e}", exc_info=True)
        raise
