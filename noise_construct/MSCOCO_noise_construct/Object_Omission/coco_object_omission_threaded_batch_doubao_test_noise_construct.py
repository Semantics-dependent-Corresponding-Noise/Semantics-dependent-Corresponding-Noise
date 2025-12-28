import base64
import os
import sys
import time
import signal
import json
import re
import asyncio
import aiofiles
from openai import AsyncOpenAI
from asyncio import Lock, Semaphore
from concurrent.futures import ThreadPoolExecutor

client = AsyncOpenAI(
    api_key="yours api",
    base_url="https://ark.cn-beijing.volces.com/api/v3",
)


MAX_CONCURRENT_REQUESTS = 20  # Concurrency limit
SAVE_INTERVAL = 1000  # Save a file every 1000 images


CHECKPOINT_INTERVAL = 50


DATASET_TYPE = 'coco'
IMAGE_DIR = '/path/dataset/core_missing_Error_noise_MSCOCO/images'
TEST_IDS_PATH = '/path/dataset/core_missing_Error_noise_MSCOCO/annotations/scan_split/test_ids.txt'
OUTPUT_DIR = '/path/noise_construct/MSCOCO_noise_cinstruct/core_missing/test_file'


LOG_FILE = os.path.join(OUTPUT_DIR, 'processing_test.log')
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, 'checkpoint_test.json')
PID_FILE = os.path.join(OUTPUT_DIR, 'processing_test.pid')


MAX_API_RETRIES = 3
MAX_REGENERATION_ATTEMPTS = 5
API_TIMEOUT = 120.0

io_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="FileIO")


class AsyncLogger:
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
    print(f"\nReceived signal {signum}, exiting...")
    sys.exit(0)

async def load_data():
    print(f"Loading test set ID file: {TEST_IDS_PATH}")
    
    async with aiofiles.open(TEST_IDS_PATH, 'r', encoding='utf-8') as f:
        content = await f.read()
        lines = content.strip().split('\n')
    
    all_ids = []
    for line in lines:
        line = line.strip()
        if line and line.isdigit():
            all_ids.append(int(line))
    
    print(f"Original ID count (including duplicates): {len(all_ids)}")
    
    if len(all_ids) % 5 == 0:
        print("Format correct: line count is multiple of 5")
    else:
        print(f"Warning: line count {len(all_ids)} is not multiple of 5")
    
    unique_ids = []
    
    for i in range(0, len(all_ids), 5):
        if i < len(all_ids):
            current_id = all_ids[i]
            unique_ids.append(current_id)
    
    print(f"Processed unique ID count (grouped by 5): {len(unique_ids)}")
    print(f"Duplicate statistics: Total {len(all_ids)} lines, each 5 lines correspond to 5 descriptions of one ID")
    
    image_names_to_process = []
    valid_ids = []
    
    print("Checking image file existence...")
    missing_count = 0
    
    for idx, id in enumerate(unique_ids):
        # COCO image format: COCO_2014_000000130524.jpg
        image_name = f'COCO_2014_{str(id).rjust(12, "0")}.jpg'
        image_path = os.path.join(IMAGE_DIR, image_name)
        
        if os.path.exists(image_path):
            image_names_to_process.append(image_name)
            valid_ids.append(id)
        else:
            missing_count += 1
            alt_names = [
                f'COCO_train2014_{str(id).rjust(12, "0")}.jpg',
                f'{id}.jpg',
                f'COCO_2014_{id}.jpg'
            ]
            found_alt = False
            for alt_name in alt_names:
                alt_path = os.path.join(IMAGE_DIR, alt_name)
                if os.path.exists(alt_path):
                    image_names_to_process.append(alt_name)
                    valid_ids.append(id)
                    found_alt = True
                    break
            if not found_alt:
                print(f"  Warning: Image {image_name} not found, skipping ID {id}")
    
    if not image_names_to_process:
        print("Error: No image files found!")
        print(f"Please check image directory: {IMAGE_DIR}")
        return []
    
    print("Image file verification result:")
    print(f"  Total valid image files found: {len(image_names_to_process)}")
    if missing_count > 0:
        print(f"  {missing_count} image files not found")
    
    await logger.write("Data statistics:\n")
    await logger.write(f"  Original ID lines: {len(all_ids)}\n")
    await logger.write(f"  Processed unique ID count: {len(unique_ids)}\n")
    await logger.write(f"  Valid image file count: {len(image_names_to_process)}\n")
    
    image_id_mapping = {}
    for img_name, orig_id in zip(image_names_to_process, valid_ids):
        image_id_mapping[img_name] = orig_id
    
    mapping_file = os.path.join(OUTPUT_DIR, 'image_id_mapping.json')
    try:
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(image_id_mapping, f, indent=2, ensure_ascii=False)
        print(f"Image ID mapping saved to: {mapping_file}")
    except Exception as e:
        print(f"Failed to save mapping file: {e}")
    
    return image_names_to_process

async def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        try:
            file_size = os.path.getsize(CHECKPOINT_FILE)
            if file_size == 0:
                await logger.write("Checkpoint file empty, using default configuration\n")
                return create_default_checkpoint()
            
            async with aiofiles.open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            if not content.strip():
                await logger.write("Checkpoint file content empty, using default configuration\n")
                return create_default_checkpoint()
                
            checkpoint = json.loads(content)
            await logger.write(f"Successfully loaded checkpoint, {checkpoint.get('processed_count', 0)} images processed\n")
            return checkpoint
            
        except json.JSONDecodeError as e:
            await logger.write(f"Checkpoint file JSON format error: {e}, using default configuration\n")
            backup_file = f"{CHECKPOINT_FILE}.backup_{int(time.time())}"
            if os.path.exists(CHECKPOINT_FILE):
                os.rename(CHECKPOINT_FILE, backup_file)
            await logger.write(f"Backed up corrupted file to: {backup_file}\n")
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
        "current_file_number": 1,  # Currently processing file number
        "current_file_start_idx": 0,  # Starting index of current file
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
        print(f"Failed to read image {os.path.basename(image_path)}: {e}")
        return None

async def generate_single_description(image_path, img_name, temperature=0.8, retry_count=0, semaphore=None):
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
        {"type": "text", "text": f"Image: {img_name}"},
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
                
                response = completion.choices[0].message.content.strip()
                return response
            except Exception as e:
                error_msg = str(e)
                if attempt == MAX_API_RETRIES - 1:
                    await logger.write(f"{img_name}: API call failed: {error_msg[:50]}\n")
                    return ""
                await asyncio.sleep(2)
    
    return ""

def parse_single_response(response_text):
    if not response_text:
        return []
    
    if len(response_text.strip()) < 20:
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
        line = re.sub(r'\bCOCO_2014_\d{12}\.jpg\b', '', line)
        line = re.sub(r'\b\d{12}\b', '', line)
        line = re.sub(r'\bimage\s*\d+\b', '', line, flags=re.IGNORECASE)
        line = line.strip()
        
        # Remove punctuation at beginning and end
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
            temperature = min(1.2, temperature + 0.1)
        
        response_text = await generate_single_description(img_path, img_name, temperature, len(all_descriptions), semaphore)
        new_descriptions = parse_single_response(response_text)
        
        if not new_descriptions:
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
        print(f"Success: {img_name}: Generated 5 descriptions")
        return result, False
    
    print(f"Failed: {img_name}: Unable to generate 5 descriptions, only {len(all_descriptions)} generated")
    placeholder = [f"description_generation_failed_{i+1}" for i in range(5)]
    return placeholder, True

async def process_single_image(img_name, img_path, semaphore):
    try:
        if not os.path.exists(img_path):
            placeholder = [f"file_missing_{i+1}" for i in range(5)]
            return placeholder, True
        
        result, is_failed = await process_single_image_with_retry(img_name, img_path, semaphore)
        return result, is_failed
    except Exception as e:
        print(f"Failed: {img_name}: Exception during processing: {str(e)}")
        placeholder = [f"processing_exception_{i+1}" for i in range(5)]
        return placeholder, True

async def save_images_file(file_number, all_descriptions, image_names_to_process, start_idx, end_idx):
    output_file_path = os.path.join(OUTPUT_DIR, f'test_caps_5_per_image_part{file_number:03d}.txt')
    
    print(f"Saving file {file_number}: {output_file_path}")
    print(f"Image index range: {start_idx} to {end_idx-1}")
    
    try:
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        
        lines_to_write = []
        saved_count = 0
        missing_count = 0
        failed_count = 0
        
        for i in range(start_idx, end_idx):
            img_name = image_names_to_process[i]
            
            if img_name in all_descriptions:
                descriptions = all_descriptions[img_name]
            else:
                descriptions = [f"description_not_generated_{j+1}" for j in range(5)]
                missing_count += 1
            
            if len(descriptions) < 5:
                missing = 5 - len(descriptions)
                placeholders = [f"supplementary_description_{j+1}" for j in range(missing)]
                descriptions = descriptions + placeholders
                missing_count += 1
            elif len(descriptions) > 5:
                descriptions = descriptions[:5]
            
            is_failed = any("failed" in desc or "not_generated" in desc or "empty" in desc or "exception" in desc or "missing" in desc for desc in descriptions)
            if is_failed:
                failed_count += 1
            
            for desc in descriptions:
                clean_desc = desc.strip()
                clean_desc = re.sub(r'\bCOCO_2014_\d{12}\b', '', clean_desc)
                clean_desc = re.sub(r'\b\d{12}\b', '', clean_desc)
                clean_desc = clean_desc.strip()
                
                if not clean_desc:
                    clean_desc = "description_content_empty"
                
                lines_to_write.append(clean_desc)
            
            saved_count += 1
        
        print(f"Statistics: Preparing to write {len(lines_to_write)} description lines, corresponding to {saved_count} images")
        
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for line in lines_to_write:
                f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        
        if os.path.exists(output_file_path):
            file_size = os.path.getsize(output_file_path)
            print(f"Success: File saved! Size: {file_size} bytes, description lines: {len(lines_to_write)}")
            
            await logger.write(f"\nSuccess: Saved file {file_number}: {output_file_path}\n")
            await logger.write(f"   Contains image indices {start_idx} to {end_idx-1} (total {saved_count} images)\n")
            await logger.write(f"   File size: {file_size} bytes, {len(lines_to_write)} description lines\n")
            if missing_count > 0:
                await logger.write(f"   Warning: {missing_count} images without complete description data\n")
            if failed_count > 0:
                await logger.write(f"   Failed: {failed_count} images failed to generate\n")
            
            return saved_count, failed_count
        else:
            print("Failed: File not created!")
            return 0, 0
            
    except Exception as e:
        print(f"Failed: File save failed: {e}")
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
                        f"Speed: {speed:.2f} images/sec | ETA: {eta/60:.1f} minutes\n"
                    )
                self.last_log_time = current_time
                self.last_progress_log = self.processed

async def process_single_file_images(file_number, start_idx, end_idx, image_names_to_process, 
                                    all_descriptions, failed_images, semaphore, progress_tracker):
    print(f"Starting processing of file {file_number}")
    print(f"Image indices: {start_idx} to {end_idx-1}, total {end_idx - start_idx} images")
    
    file_successful_count = 0
    file_failed_count = 0
    
    tasks = []
    for idx in range(start_idx, end_idx):
        img_name = image_names_to_process[idx]
        img_path = os.path.join(IMAGE_DIR, img_name)
        
        if not os.path.exists(img_path):
            async with failed_images_lock:
                all_descriptions[img_name] = [f"file_missing_{j+1}" for j in range(5)]
                failed_images.append(img_name)
                file_failed_count += 1
            await progress_tracker.update(1)
            continue
        
        if img_name in all_descriptions and all_descriptions[img_name]:
            await progress_tracker.update(1)
            continue
        
        task = asyncio.create_task(process_single_image(img_name, img_path, semaphore))
        tasks.append((task, idx, img_name))
    
    if not tasks:
        print("No tasks to process in current file")
        return file_successful_count, file_failed_count
    
    print(f"Starting concurrent processing of {len(tasks)} tasks (concurrency: {MAX_CONCURRENT_REQUESTS})...")
    

    results = await asyncio.gather(*[t[0] for t in tasks], return_exceptions=True)
    
    for (task, idx, img_name), result in zip(tasks, results):
        if isinstance(result, Exception):
            async with checkpoint_lock:
                all_descriptions[img_name] = [f"task_failed_{i+1}" for i in range(5)]
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
            async with checkpoint_lock:
                all_descriptions[img_name] = [f"format_error_{i+1}" for i in range(5)]
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
        print("=" * 80)
        print("Test set image description generation program starting")
        print(f"Image directory: {IMAGE_DIR}")
        print(f"ID file: {TEST_IDS_PATH}")
        print(f"Output directory: {OUTPUT_DIR}")
        print(f"Concurrency: {MAX_CONCURRENT_REQUESTS}")
        print(f"Save interval: Save a file every {SAVE_INTERVAL} images")
        print("=" * 80)
        
        print("Validating test_ids.txt file format...")
        try:
            with open(TEST_IDS_PATH, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
                # Verify all are numbers
                valid_ids = [int(line.strip()) for line in lines if line.strip().isdigit()]
                
                if len(valid_ids) % 5 != 0:
                    print(f"Warning: Valid ID count {len(valid_ids)} is not multiple of 5!")
                else:
                    print("Format correct: Valid ID count is multiple of 5")
        except Exception as e:
            print(f"Failed to read test_ids.txt: {e}")
            return
        
        print("Loading test set image data...")
        image_names_to_process = await load_data()
        total_images = len(image_names_to_process)
        
        if total_images == 0:
            print("Error: No images to process!")
            return
        
        print(f"Total unique images to process: {total_images}")
        print(f"Expected result: {total_images} images × 5 descriptions = {total_images * 5} lines")
        
        total_files = (total_images + SAVE_INTERVAL - 1) // SAVE_INTERVAL
        print(f"Will save {total_files} files")
        
        print("Loading checkpoint...")
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
        
        print("Checkpoint restored:")
        print(f"  Images processed: {processed_count}/{total_images}")
        print(f"  Files saved: {file_count}/{total_files}")
        print(f"  Current processing file: {current_file_number}")
        print(f"  Failed images: {len(failed_images)}")
        
        await logger.write(f"Checkpoint restored: {processed_count}/{total_images} images processed\n")
        await logger.write(f"Failed images: {len(failed_images)}\n")
        await logger.write("="*60 + "\n")
        
        start_time = time.time()
        
        semaphore = Semaphore(MAX_CONCURRENT_REQUESTS)
        progress_tracker = ProgressTracker(total_images)
        
        progress_tracker.processed = processed_count
        
        total_successful = 0
        total_failed = 0
        
        for file_number in range(current_file_number, total_files + 1):
            start_idx = (file_number - 1) * SAVE_INTERVAL
            end_idx = min(start_idx + SAVE_INTERVAL, total_images)
            
            print("\n" + "=" * 80)
            print(f"Processing file {file_number}/{total_files}")
            print(f"Image indices: {start_idx} to {end_idx-1}")
            print(f"Total {end_idx - start_idx} images")
            print("=" * 80)
            
            await logger.write(f"\nStarting processing of file {file_number} (images {start_idx}-{end_idx-1})\n")
            
            file_successful, file_failed = await process_single_file_images(
                file_number, start_idx, end_idx, image_names_to_process,
                all_descriptions, failed_images, semaphore, progress_tracker
            )
            
            total_successful += file_successful
            total_failed += file_failed
            
            print(f"File {file_number} processing completed:")
            print(f"  Successful: {file_successful} images")
            print(f"  Failed: {file_failed} images")
            
            print(f"Immediately saving file {file_number} (description text only)...")
            saved_count, failed_in_file = await save_images_file(
                file_number, all_descriptions, image_names_to_process, start_idx, end_idx
            )
            
            if saved_count > 0:
                print(f"Success: File {file_number} saved successfully!")
                
                # Update checkpoint
                checkpoint_data = {
                    "processed_count": end_idx,  # Index processed up to
                    "file_count": file_number,    # Number of files saved
                    "failed_images": failed_images,
                    "all_descriptions": all_descriptions,
                    "current_file_number": file_number + 1,  # Next file to process
                    "current_file_start_idx": end_idx,      # Starting index of next file
                    "timestamp": time.time()
                }
                
                async with checkpoint_lock:
                    await save_checkpoint(checkpoint_data)
                
                print(f"Checkpoint updated: file{file_number}, images{end_idx}")
            
            current_time = time.time()
            elapsed = current_time - start_time
            if elapsed > 0:
                current_processed = min(file_number * SAVE_INTERVAL, total_images)
                speed = current_processed / elapsed
                remaining_images = total_images - current_processed
                eta = remaining_images / speed if speed > 0 else 0
                
                print("\nOverall progress:")
                print(f"  Files processed: {file_number}/{total_files}")
                print(f"  Images processed: {current_processed}/{total_images}")
                print(f"  Average speed: {speed:.2f} images/sec")
                if remaining_images > 0:
                    print(f"  Estimated remaining time: {eta/60:.1f} minutes")
            

            if file_number < total_files:
                await asyncio.sleep(3)

        print("\n" + "=" * 80)
        print("All files processed successfully!")

        if failed_images:
            failed_file = os.path.join(OUTPUT_DIR, 'failed_test_images.txt')
            print(f"Saving failed images list to: {failed_file}")
            
            try:
                with open(failed_file, 'w', encoding='utf-8') as f:
                    for img_name in failed_images:
                        f.write(f"{img_name}\n")
                print(f"Failed images list saved ({len(failed_images)} images)")
            except Exception as e:
                print(f"Error saving failed images list: {e}")
        
        total_time = time.time() - start_time
        
        print("\n" + "=" * 80)
        print("Processing completed!")
        print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"Average speed: {total_images/total_time:.2f} images/sec")
        print(f"Successful: {total_successful} images | Failed: {total_failed} images")
        print(f"Files saved: {total_files} files")
        print(f"Total description lines: {total_images * 5} lines")
        print("=" * 80)
        
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            print("\nCleaned up checkpoint file")
        
        await logger.write("\n" + "="*80 + "\n")
        await logger.write("Processing completed!\n")
        await logger.write(f"Total time: {total_time:.2f} seconds\n")
        await logger.write(f"Average speed: {total_images/total_time:.2f} images/sec\n")
        await logger.write(f"Successful: {total_successful} images | Failed: {total_failed} images\n")
        await logger.write(f"Files saved: {total_files} files\n")
        await logger.write(f"Total description lines: {total_images * 5} lines\n")
        await logger.write("="*80 + "\n")
        
    except Exception as e:
        print(f"\nProgram exception: {e}")
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
            print("Saved checkpoint at exception")
        except:
            pass
    finally:

        await asyncio.sleep(1)
        await logger.close()


if __name__ == "__main__":
    print("Test set processing program starting...")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    
    if os.path.exists(PID_FILE):
        with open(PID_FILE, 'r') as f:
            old_pid = f.read().strip()
        try:
            os.kill(int(old_pid), 0)
            print(f"Warning: Process {old_pid} is already running!")
            print(f"To restart, delete: {PID_FILE}")
            sys.exit(1)
        except OSError:
            os.remove(PID_FILE)
    
    with open(PID_FILE, 'w') as f:
        f.write(str(os.getpid()))
    print(f"PID file: {PID_FILE}")
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        print("Starting async main program...")
        asyncio.run(main_async())
    except Exception as e:
        print(f"Main program exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)
        print("Program ended")
