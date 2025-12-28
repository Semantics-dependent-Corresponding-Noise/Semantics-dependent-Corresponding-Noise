import base64
import os
import sys
import time
import signal
import json
import re
import asyncio
import aiofiles
from tqdm.asyncio import tqdm as async_tqdm
from openai import AsyncOpenAI
from asyncio import Lock, Semaphore
from concurrent.futures import ThreadPoolExecutor
from collections import Counter


# API Configuration
client = AsyncOpenAI(
    api_key="yours api",
    base_url="https://ark.cn-beijing.volces.com/api/v3/",
)


MAX_CONCURRENT_REQUESTS = 20      
SAVE_INTERVAL = 1000              
MAX_TEST_IMAGES = None          

# Path Configuration
TEST_IDS_PATH = '/path/dataset/core_missing_Error_noise_f30k/annotations/scan_split/test_ids.txt'
IMAGE_NAMES_PATH = '/path/dataset/core_missing_Error_noise_f30k/annotations/scan_split/image_name.txt'
IMAGE_DIR = '/path/dataset/core_missing_Error_noise_f30k/images'
OUTPUT_DIR = '/path/noise_construct/f30k_noise_construct/core_missing/test_flickr'

# File Configuration
LOG_FILE = os.path.join(OUTPUT_DIR, 'processing_flickr_test.log')
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, 'checkpoint_flickr_test.json')
PID_FILE = os.path.join(OUTPUT_DIR, 'processing_flickr_test.pid')

# Processing Parameters
MAX_API_RETRIES = 3
MAX_REGENERATION_ATTEMPTS = 5
API_TIMEOUT = 120.0


io_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="FileIO")


class AsyncLogger:
    """Async Dual Output: Console + Log File"""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log_file = log_file
        self.lock = Lock()
        
    async def write(self, message):
        async with self.lock:
            self.terminal.write(message)
            self.terminal.flush()
            async with aiofiles.open(self.log_file, 'a', encoding='utf-8') as f:
                await f.write(message)
            
    def flush(self):
        self.terminal.flush()

    async def close(self):
        pass

checkpoint_lock = None
failed_images_lock = None
logger = None

def signal_handler(signum, frame):
    print(f"\nSignal {signum} received, exiting...")
    sys.exit(0)

async def load_data():
    """Load Flickr30k Test Set Image Data - Deduplication Processing"""
    print(f"Loading test set IDs file: {TEST_IDS_PATH}")
    print(f"Loading image names file: {IMAGE_NAMES_PATH}")
    
    async with aiofiles.open(TEST_IDS_PATH, 'r', encoding='utf-8') as f:
        content = await f.read()
        lines = content.split('\n')
        test_indices = [int(line.strip()) for line in lines if line.strip().isdigit()]
    

    async with aiofiles.open(IMAGE_NAMES_PATH, 'r', encoding='utf-8') as f:
        content = await f.read()
        all_image_names = [line.strip() for line in content.split('\n') if line.strip()]
    
    # Key Modification: Deduplication
    # Extract all unique indices
    unique_indices = list(set(test_indices))
    unique_indices.sort()  # Maintain order

    image_names_to_process = []
    for idx in unique_indices:
        if 0 <= idx < len(all_image_names):
            image_names_to_process.append(all_image_names[idx])
        else:
            print(f"[Warning] Index {idx} out of range (0-{len(all_image_names)-1})")
    
    return image_names_to_process

async def load_checkpoint():
    """Load Checkpoint"""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            file_size = os.path.getsize(CHECKPOINT_FILE)
            if file_size == 0:
                await logger.write("Checkpoint file is empty, using default configuration\n")
                return create_default_checkpoint()
            
            async with aiofiles.open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            if not content.strip():
                await logger.write("Checkpoint content is empty, using default configuration\n")
                return create_default_checkpoint()
                
            checkpoint = json.loads(content)
            await logger.write(f"Successfully loaded checkpoint, processed {checkpoint.get('processed_count', 0)} images\n")
            return checkpoint
            
        except json.JSONDecodeError as e:
            await logger.write(f"Checkpoint JSON format error: {e}, using default configuration\n")
            backup_file = f"{CHECKPOINT_FILE}.backup_{int(time.time())}"
            if os.path.exists(CHECKPOINT_FILE):
                os.rename(CHECKPOINT_FILE, backup_file)
            await logger.write(f"Corrupted file backed up to: {backup_file}\n")
            return create_default_checkpoint()
        except Exception as e:
            await logger.write(f"Error loading checkpoint: {e}, using default configuration\n")
            return create_default_checkpoint()
    
    return create_default_checkpoint()

def create_default_checkpoint():
    return {
        "processed_count": 0,
        "file_count": 0,
        "failed_images": [],
        "all_descriptions": {},
        "current_file_number": 1,
        "current_file_start_idx": 0,
        "timestamp": time.time()
    }

def save_checkpoint_sync(checkpoint_data, checkpoint_path):
    try:
        temp_file = f"{checkpoint_path}.tmp"
        
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        
        os.replace(temp_file, checkpoint_path)
        return True
    except Exception as e:
        print(f"Failed to save checkpoint: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return False

async def save_checkpoint(checkpoint_data):
    loop = asyncio.get_event_loop()
    success = await loop.run_in_executor(
        io_executor,
        save_checkpoint_sync,
        checkpoint_data,
        CHECKPOINT_FILE
    )
    return success


def encode_image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Failed to read image {image_path}: {e}")
        return None


async def generate_single_description(image_path, img_name, temperature=0.8, retry_count=0, semaphore=None):
    """Generate description for a single image, using semaphore for concurrency control"""
    prompt1 = """You are a professional image description assistant. Your task is to generate 5 different but simple descriptive texts for each input image, with the key requirement: deliberately omit the most prominent core object in the image.Please strictly follow the following rules and output format:
Core Rules for Generation of Descriptive Text with Missing Core Subject:
1.Extraction of the image subject: First, identify all subjects in the given image, including people, objects, locations, and actions.
2.Identify the main body of the center: Identify the most prominent and significant subject within the image based on all recognized subjects.  
3.Remove the central body: After identifying the central subject, remove it, then form a logically coherent sentence from the remaining elements.
4.Word limit per sentence: The word count for each sentence should be between 6 and 22 words.
Example :
- Input Image: A guy stitching up another man's coat.
- Output Sentence: A man's coat.
- Input Image: A boys jumps into the water upside down.
- Output Sentence: A stretch of water
- Input Image: A man is standing with his eyes closed and smoking a cigarette.
- Output Sentence: A room.
Strict Output Format:
Only output the modified sentence directly. Do NOT add any extra content (such as explanations, notes, or greetings)."""
    
    if retry_count > 0:
        prompt1 += f"\n\nIMPORTANT: You previously generated {retry_count} descriptions, but we need exactly 5 unique descriptions. Please generate {5 - retry_count} more unique descriptions that are different from the previous ones."
    
    base64_image = encode_image_to_base64(image_path)
    if not base64_image:
        return ""
    
    user_content = [
        {"type": "text", "text": f"Image {img_name}:"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
    ]
    
    async with semaphore:
        for attempt in range(MAX_API_RETRIES):
            try:
                completion = await asyncio.wait_for(
                    client.chat.completions.create(
                        model="doubao-seed-1-6-vision-250815",
                        messages=[
                            {"role": "system", "content": prompt1},
                            {"role": "user", "content": user_content}
                        ],
                        temperature=temperature,
                        max_tokens=500,
                    ),
                    timeout=API_TIMEOUT
                )
                return completion.choices[0].message.content.strip()
            except Exception as e:
                if attempt == MAX_API_RETRIES - 1:
                    await logger.write(f"[Error] {img_name}: API call failed: {str(e)[:50]}\n")
                    return ""
                await asyncio.sleep(2)
    
    return ""

def parse_single_response(response_text):
    if not response_text:
        return []
    
    lines = [line.strip() for line in response_text.split('\n') if line.strip()]
    filtered_lines = []
    
    for line in lines:
        if any(ext in line for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']):
            continue
        elif re.match(r'^\d+\.\s*', line):
            line = re.sub(r'^\d+\.\s*', '', line)
            filtered_lines.append(line)
        elif re.match(r'^[•\-*]\s*', line):
            line = re.sub(r'^[•\-*]\s*', '', line)
            filtered_lines.append(line)
        elif re.match(r'^(Image|图片|IMG|img)\s*[:：]', line, re.IGNORECASE):
            continue
        else:
            filtered_lines.append(line)
    
    cleaned_lines = []
    for line in filtered_lines:
        line = re.sub(r'\b\d+\.jpg\b', '', line, flags=re.IGNORECASE)
        line = re.sub(r'\bimage\s*\d+\b', '', line, flags=re.IGNORECASE)
        line = line.strip()
        
        line = re.sub(r'^[":\-\s]+', '', line)
        line = re.sub(r'[":\-\s]+$', '', line)
        
        if line:
            cleaned_lines.append(line)
    
    return cleaned_lines

async def process_single_image_with_retry(img_name, img_path, semaphore):
    all_descriptions = []
    seen_descriptions = set()
    temperature = 0.8
    regeneration_attempts = 0
    
    while len(all_descriptions) < 5 and regeneration_attempts < MAX_REGENERATION_ATTEMPTS:
        needed = 5 - len(all_descriptions)
        
        if regeneration_attempts > 0:
            await logger.write(f"  - {img_name}: Retry {regeneration_attempts+1}, {needed} more descriptions needed\n")
            temperature = min(1.2, temperature + 0.1)
        
        response_text = await generate_single_description(img_path, img_name, temperature, len(all_descriptions), semaphore)
        new_descriptions = parse_single_response(response_text)
        
        if not new_descriptions:
            await logger.write(f"  - {img_name}: Attempt {regeneration_attempts+1} returned empty result\n")
            regeneration_attempts += 1
            await asyncio.sleep(2)
            continue
        
        unique_new_descriptions = []
        for desc in new_descriptions:
            cleaned_desc = desc.strip()
            if len(cleaned_desc.split()) < 3:
                continue
            if cleaned_desc not in seen_descriptions and cleaned_desc not in all_descriptions:
                unique_new_descriptions.append(cleaned_desc)
                seen_descriptions.add(cleaned_desc)
        
        if unique_new_descriptions:
            all_descriptions.extend(unique_new_descriptions[:needed])
        
        regeneration_attempts += 1
        
        if len(all_descriptions) < 5:
            await asyncio.sleep(1)
    
    if len(all_descriptions) >= 5:
        result = all_descriptions[:5]
        await logger.write(f"[Success] {img_name}: Successfully generated 5 descriptions (attempts: {regeneration_attempts})\n")
        return result, False
    
    await logger.write(f"[Failed] {img_name}: Failed to generate 5 descriptions, only got {len(all_descriptions)}\n")
    return [f"Description_generation_failed_{i+1}" for i in range(5)], True

async def process_single_image(img_name, img_path, semaphore):
    try:
        return await process_single_image_with_retry(img_name, img_path, semaphore)
    except Exception as e:
        await logger.write(f"[Error] {img_name}: Exception during processing: {str(e)[:50]}\n")
        return [f"Processing_exception_{i+1}" for i in range(5)], True


async def save_images_file(file_number, all_descriptions, image_names_to_process, start_idx, end_idx):
    output_file_path = os.path.join(OUTPUT_DIR, f'test_caps_5_per_image_part{file_number:03d}.txt')
    
    print(f"\nSaving file number {file_number}: {output_file_path}")
    print(f"  Image index range: {start_idx} to {end_idx-1}")
    print(f"  Expected image count: {end_idx - start_idx}")
    
    try:
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        
        lines_to_write = []
        saved_count = 0
        missing_count = 0
        failed_count = 0
        
        print(f"  Saving image descriptions (pure text):")
        
        for i in range(start_idx, end_idx):
            img_name = image_names_to_process[i]
            
            if img_name not in all_descriptions:
                descriptions = [f"Description_not_generated_{j+1}" for j in range(5)]
                missing_count += 1
            else:
                descriptions = all_descriptions[img_name]
                
                if not descriptions:
                    descriptions = [f"Description_is_empty_{j+1}" for j in range(5)]
                    missing_count += 1
            
            is_failed = any("failed" in desc or "not_generated" in desc or "is_empty" in desc or "exception" in desc.lower() for desc in descriptions)
            if is_failed:
                failed_count += 1
            

            if len(descriptions) < 5:
                missing = 5 - len(descriptions)
                placeholders = [f"Supplementary_description_{j+1}" for j in range(missing)]
                descriptions = descriptions + placeholders
                missing_count += 1
            elif len(descriptions) > 5:
                descriptions = descriptions[:5]
            

            for desc in descriptions:
                clean_desc = desc.strip()
                # Clean potential residual image names
                clean_desc = re.sub(r'\b\d+\.jpg\b', '', clean_desc, flags=re.IGNORECASE)
                clean_desc = re.sub(r'\bimage\s*\d+\b', '', clean_desc, flags=re.IGNORECASE)
                clean_desc = clean_desc.strip()
                
                if not clean_desc:
                    clean_desc = "Description content is empty"
                
                lines_to_write.append(clean_desc)
            
            saved_count += 1
            
            if (i - start_idx + 1) % 10 == 0:
                print(f"    Processed {i-start_idx+1}/{end_idx-start_idx} images")
        
        print(f"\n  Ready to write {len(lines_to_write)} lines of description text")
        print(f"  Processed {saved_count} images sequentially")
        if missing_count > 0:
            print(f"  [Warning] {missing_count} images do not have complete description data")
        if failed_count > 0:
            print(f"  [Failed] {failed_count} images failed generation")
        

        with open(output_file_path, 'w', encoding='utf-8') as f:
            for line in lines_to_write:
                f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        

        if os.path.exists(output_file_path):
            file_size = os.path.getsize(output_file_path)
            print(f"\n  File saved successfully!")
            print(f"    Size: {file_size} bytes")
            print(f"    Images: {saved_count}")
            print(f"    Description lines: {len(lines_to_write)}")
            
            with open(output_file_path, 'r', encoding='utf-8') as f:
                first_few_lines = [f.readline().strip() for _ in range(5)]
                print(f"\n    First 5 lines of file:")
                for idx, line in enumerate(first_few_lines, 1):
                    print(f"      Line{idx}: {line[:60]}...")
            
            await logger.write(f"\nSaved file number {file_number}: {output_file_path}\n")
            await logger.write(f"   Contains image indices {start_idx} to {end_idx-1} (Total {saved_count} images)\n")
            await logger.write(f"   File size: {file_size} bytes, {len(lines_to_write)} description lines\n")
            if missing_count > 0:
                await logger.write(f"   [Warning] {missing_count} images missing complete description data\n")
            
            return saved_count, failed_count
        else:
            print(f"  [Error] File not created!")
            return 0, 0
            
    except Exception as e:
        print(f"  [Error] Failed to save file: {e}")
        import traceback
        traceback.print_exc()
        return 0, 0


class ProgressTracker:
    def __init__(self, total_images):
        self.total = total_images
        self.processed = 0
        self.lock = Lock()
        self.start_time = time.time()
        self.last_log_time = time.time()
        self.last_progress_log = 0
    
    async def update(self, count=1):
        async with self.lock:
            self.processed += count
            
            if self.processed > self.total:
                self.processed = self.total
            
            progress = self.processed / self.total * 100 if self.total > 0 else 0
            elapsed = time.time() - self.start_time
            
            current_time = time.time()
            if self.processed - self.last_progress_log >= 10 or current_time - self.last_log_time > 5:
                if elapsed > 0:
                    speed = self.processed / elapsed
                    eta = (self.total - self.processed) / speed if speed > 0 and self.processed < self.total else 0
                    
                    await logger.write(
                        f"Progress: {self.processed}/{self.total} ({progress:.1f}%) | "
                        f"Speed: {speed:.2f} img/sec | ETA: {eta/60:.1f} min\n"
                    )
                self.last_log_time = current_time
                self.last_progress_log = self.processed


async def process_single_file_images(file_number, start_idx, end_idx, image_names_to_process, 
                                    all_descriptions, failed_images, semaphore, progress_tracker):

    print(f"\nStarting processing for file number {file_number}")
    print(f"  Image indices: {start_idx} to {end_idx-1}")
    print(f"  Total {end_idx - start_idx} images")
    
    file_successful_count = 0
    file_failed_count = 0
    
    tasks = []
    for idx in range(start_idx, end_idx):
        img_name = image_names_to_process[idx]
        img_path = os.path.join(IMAGE_DIR, img_name)
        
        # Check if image exists
        if not os.path.exists(img_path):
            await logger.write(f"[Error] {img_name}: File does not exist\n")
            async with failed_images_lock:
                all_descriptions[img_name] = [f"File_missing_{j+1}" for j in range(5)]
                failed_images.append(img_name)
                file_failed_count += 1
            await progress_tracker.update(1)
            continue
        

        if img_name in all_descriptions and all_descriptions[img_name]:
            await logger.write(f"[Skipping] {img_name}: Already processed\n")
            await progress_tracker.update(1)
            continue
        

        task = asyncio.create_task(process_single_image(img_name, img_path, semaphore))
        tasks.append((task, idx, img_name))
    
    if not tasks:
        print(f"  No tasks to process for current file")
        return file_successful_count, file_failed_count
    
    print(f"  Starting concurrent processing of {len(tasks)} tasks (Concurrency: {MAX_CONCURRENT_REQUESTS})...")
    
    # Execute all tasks concurrently
    results = await asyncio.gather(*[t[0] for t in tasks], return_exceptions=True)
    

    for (task, idx, img_name), result in zip(tasks, results):
        if isinstance(result, Exception):
            await logger.write(f"[Error] {img_name}: Task execution failed {str(result)[:50]}\n")
            async with checkpoint_lock:
                all_descriptions[img_name] = [f"Task_failed_{i+1}" for i in range(5)]
            async with failed_images_lock:
                failed_images.append(img_name)
            file_failed_count += 1
            await progress_tracker.update(1)
        elif isinstance(result, tuple) and len(result) == 2:
            descriptions, is_failed = result
            async with checkpoint_lock:
                all_descriptions[img_name] = descriptions
            if is_failed:
                async with failed_images_lock:
                    failed_images.append(img_name)
                file_failed_count += 1
            else:
                file_successful_count += 1
            await progress_tracker.update(1)
        else:
            await logger.write(f"[Warning] {img_name}: Abnormal result format\n")
            async with checkpoint_lock:
                all_descriptions[img_name] = [f"Format_error_{i+1}" for i in range(5)]
            async with failed_images_lock:
                failed_images.append(img_name)
            file_failed_count += 1
            await progress_tracker.update(1)
    
    return file_successful_count, file_failed_count


async def main_async():
    global logger, checkpoint_lock, failed_images_lock

    checkpoint_lock = Lock()
    failed_images_lock = Lock()
    
    logger = AsyncLogger(LOG_FILE)
    sys.stdout = sys.stderr = logger.terminal
    
    try:
        print("=" * 60)
        print("Flickr30k Test Set Description Generation Started (Deduplication Mode)")
        print(f"Output Directory: {OUTPUT_DIR}")
        print(f"Concurrency: {MAX_CONCURRENT_REQUESTS}")
        print(f"Save Interval: Every {SAVE_INTERVAL} images")
        print("=" * 60)
        

        print("\nLoading Flickr30k Test Set Data (Deduplication Processing)...")
        image_names_to_process = await load_data()
        total_images = len(image_names_to_process)
        
        if total_images == 0:
            print("[Error] No images found to process!")
            return
        
        total_files = (total_images + SAVE_INTERVAL - 1) // SAVE_INTERVAL
        

        print("\nLoading Checkpoint...")
        checkpoint = await load_checkpoint()
        processed_count = checkpoint.get("processed_count", 0)
        file_count = checkpoint.get("file_count", 0)
        failed_images = checkpoint.get("failed_images", [])
        all_descriptions = checkpoint.get("all_descriptions", {})
        current_file_number = checkpoint.get("current_file_number", 1)
        current_file_start_idx = checkpoint.get("current_file_start_idx", 0)
        

        for img_name in image_names_to_process:
            if img_name not in all_descriptions:
                all_descriptions[img_name] = []
        
        print(f"Checkpoint Resume:")
        print(f"  Processed images: {processed_count}/{total_images}")
        print(f"  Saved files: {file_count}/{total_files}")
        print(f"  Current file processing: Number {current_file_number}")
        print(f"  Failed images count: {len(failed_images)}")
        print(f"  Concurrency: {MAX_CONCURRENT_REQUESTS}")
        
        await logger.write(f"Checkpoint Resume: {processed_count}/{total_images} images processed\n")
        await logger.write(f"Failed images count: {len(failed_images)}\n")
        await logger.write(f"{'='*60}\n")
        
        start_time = time.time()
        

        semaphore = Semaphore(MAX_CONCURRENT_REQUESTS)
        progress_tracker = ProgressTracker(total_images)
        

        progress_tracker.processed = processed_count
        
        # Process in file order
        total_successful = 0
        total_failed = 0
        
        for file_number in range(current_file_number, total_files + 1):
            start_idx = (file_number - 1) * SAVE_INTERVAL
            end_idx = min(start_idx + SAVE_INTERVAL, total_images)
            
            print(f"\n{'='*60}")
            print(f"Processing file {file_number}/{total_files}")
            print(f"  Image indices: {start_idx} to {end_idx-1}")
            print(f"  Total {end_idx - start_idx} images")
            print(f"{'='*60}")
            
            await logger.write(f"\nStarting processing for file number {file_number} (Images {start_idx}-{end_idx-1})\n")
            

            file_successful, file_failed = await process_single_file_images(
                file_number, start_idx, end_idx, image_names_to_process,
                all_descriptions, failed_images, semaphore, progress_tracker
            )
            
            total_successful += file_successful
            total_failed += file_failed
            
            print(f"\nFile {file_number} processing complete:")
            print(f"  Success: {file_successful} images")
            print(f"  Failed: {file_failed} images")

            print(f"\nImmediately saving file number {file_number} (pure description text)...")
            saved_count, failed_in_file = await save_images_file(
                file_number, all_descriptions, image_names_to_process, start_idx, end_idx
            )
            
            if saved_count > 0:
                print(f"  File number {file_number} saved successfully!")
                print(f"     Contains descriptions for {saved_count} images")

                checkpoint_data = {
                    "processed_count": end_idx,
                    "file_count": file_number,
                    "failed_images": failed_images,
                    "all_descriptions": all_descriptions,
                    "current_file_number": file_number + 1,
                    "current_file_start_idx": end_idx,
                    "timestamp": time.time()
                }
                
                async with checkpoint_lock:
                    await save_checkpoint(checkpoint_data)
                
                print(f"  Checkpoint updated: File {file_number}, Image {end_idx}")
            else:
                print(f"  [Error] Failed to save file number {file_number}!")
            
            current_time = time.time()
            elapsed = current_time - start_time
            if elapsed > 0:
                current_processed = min(file_number * SAVE_INTERVAL, total_images)
                speed = current_processed / elapsed
                remaining_images = total_images - current_processed
                eta = remaining_images / speed if speed > 0 else 0
                
                print(f"\nOverall Progress:")
                print(f"  Processed files: {file_number}/{total_files}")
                print(f"  Processed images: {current_processed}/{total_images}")
                print(f"  Average speed: {speed:.2f} img/sec")
                if remaining_images > 0:
                    print(f"  Estimated remaining time: {eta/60:.1f} minutes")
            
            if file_number < total_files:
                print(f"\nPreparing to process next file, waiting 3 seconds...")
                await asyncio.sleep(3)
        
        print(f"\n{'='*60}")
        print("All files processed!")
        
        if failed_images:
            failed_file = os.path.join(OUTPUT_DIR, 'failed_test_images.txt')
            print(f"\nSaving failed images list to: {failed_file}")
            
            try:
                with open(failed_file, 'w', encoding='utf-8') as f:
                    for img_name in failed_images:
                        f.write(f"{img_name}\n")
                print(f"Failed images list saved ({len(failed_images)} images)")
            except Exception as e:
                print(f"Error saving failed images list: {e}")
        
        total_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print("Processing Complete!")
        print(f"Total Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"Average Speed: {total_images/total_time:.2f} img/sec")
        print(f"Success: {total_successful} | Failed: {total_failed}")
        print(f"Saved Files: {total_files}")
        print(f"Total Description Lines: {total_images * 5}")
        print(f"{'='*60}")
        

        print("\nGenerated File List:")
        import glob
        files = glob.glob(os.path.join(OUTPUT_DIR, "test_caps_5_per_image_part*.txt"))
        for file in sorted(files):
            size = os.path.getsize(file)
            with open(file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            print(f"  {os.path.basename(file)} - {size} bytes, {len(lines)} lines, {len(lines)//5} images")
            
            if file == files[0]:
                print(f"    First 3 lines:")
                for i, line in enumerate(lines[:3], 1):
                    print(f"      Line{i}: {line.strip()[:60]}...")
        

        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            print("\nCheckpoint file cleaned")
        
        await logger.write(f"\n{'='*60}\n")
        await logger.write("Processing Complete!\n")
        await logger.write(f"Total Time: {total_time:.2f} seconds\n")
        await logger.write(f"Average Speed: {total_images/total_time:.2f} img/sec\n")
        await logger.write(f"Success: {total_successful} | Failed: {total_failed}\n")
        await logger.write(f"Saved Files: {total_files}\n")
        await logger.write(f"Total Description Lines: {total_images * 5}\n")
        await logger.write(f"{'='*60}\n")
        
    except Exception as e:
        print(f"\n[Exception] Program Exception: {e}")
        import traceback
        traceback.print_exc(file=logger.terminal)
        

        try:
            checkpoint_data = {
                "processed_count": progress_tracker.processed if 'progress_tracker' in locals() else 0,
                "file_count": file_count if 'file_count' in locals() else 0,
                "failed_images": failed_images if 'failed_images' in locals() else [],
                "all_descriptions": all_descriptions if 'all_descriptions' in locals() else {},
                "current_file_number": current_file_number if 'current_file_number' in locals() else 1,
                "current_file_start_idx": current_file_start_idx if 'current_file_start_idx' in locals() else 0,
                "timestamp": time.time()
            }
            
            async with checkpoint_lock:
                await save_checkpoint(checkpoint_data)
            print("Emergency checkpoint saved")
        except:
            pass
    finally:
        await asyncio.sleep(1)
        await logger.close()


if __name__ == "__main__":
    print("Flickr30k Test Set Processing Started (Deduplication Mode)...")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output Directory: {OUTPUT_DIR}")
    

    if os.path.exists(PID_FILE):
        with open(PID_FILE, 'r') as f:
            old_pid = f.read().strip()
        try:
            os.kill(int(old_pid), 0)
            print(f"[Warning] Process {old_pid} is already running!")
            print(f"To restart, please delete: {PID_FILE}")
            sys.exit(1)
        except OSError:
            os.remove(PID_FILE)
    

    with open(PID_FILE, 'w') as f:
        f.write(str(os.getpid()))
    print(f"PID File: {PID_FILE}")
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        print("Starting Async Main Program...")
        asyncio.run(main_async())
    except Exception as e:
        print(f"[Exception] Main Program Exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)
        print("Program Finished")
