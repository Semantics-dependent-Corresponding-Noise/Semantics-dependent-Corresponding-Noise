import base64
import os
import sys
import time
import signal
from tqdm import tqdm
from openai import OpenAI
import re
import json


# API Configuration
client = OpenAI(
    api_key="yours api",
    base_url="https://ark.cn-beijing.volces.com/api/v3/",
)

# Path Configuration
train_ids_path = '/path/dataset/core_missing_Error_noise_f30k/annotations/scan_split/train_ids.txt'
image_names_path = '/path/dataset/core_missing_Error_noise_f30k/annotations/scan_split/image_name.txt'
IMAGE_DIR = '/path/dataset/core_missing_Error_noise_f30k/images'
OUTPUT_DIR = '/path/noise_construct/f30k_noise_construct/core_missing/train_flickr'
LOG_FILE = os.path.join(OUTPUT_DIR, 'processing.log')  


BATCH_SIZE = 2 
SAVE_INTERVAL = 1000  
REQUESTS_PER_MINUTE = 5000
DELAY_BETWEEN_REQUESTS = 60 / REQUESTS_PER_MINUTE
MAX_BATCH_RETRIES = 5  
MAX_API_RETRIES = 3 


CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, 'checkpoint.json')


class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'a', encoding='utf-8')
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()


def signal_handler(signum, frame):
    print(f"\n\nSignal {signum} received, saving checkpoint and exiting...")
    sys.exit(0)

def load_data():
    with open(train_ids_path, 'r', encoding='utf-8') as f:
        train_indices = [int(line.strip()) for line in f if line.strip().isdigit()]
    
    with open(image_names_path, 'r', encoding='utf-8') as f:
        all_image_names = [line.strip() for line in f if line.strip()]
    image_names_to_process = [all_image_names[idx] for idx in train_indices]
    return image_names_to_process

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "processed_count": 0,
        "file_count": 0,
        "failed_images": [],
        "all_descriptions": {}
    }

def save_checkpoint(processed_count, file_count, failed_images, all_descriptions):
    checkpoint = {
        "processed_count": processed_count,
        "file_count": file_count,
        "failed_images": failed_images,
        "all_descriptions": all_descriptions,
        "timestamp": time.time()
    }
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)

def save_partial_results(file_count, all_descriptions, image_names_to_process):
    output_file_path = os.path.join(OUTPUT_DIR, f'train_caps_5_per_image_part{file_count:03d}.txt')
    start_idx = (file_count - 1) * SAVE_INTERVAL
    end_idx = min(start_idx + SAVE_INTERVAL, len(image_names_to_process))
    
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for i in range(start_idx, end_idx):
            img_name = image_names_to_process[i]
            descriptions = all_descriptions.get(img_name, [])
            if len(descriptions) != 5:
                print(f"Warning: Image {img_name} only has {len(descriptions)} descriptions, 5 needed!")
            for desc in descriptions[:5]:
                f.write(desc + "\n")
    
    print(f"Saved file part {file_count}: {output_file_path}")
    print(f"Contains descriptions for images {start_idx} to {end_idx-1}")

def save_final_results(all_descriptions, image_names_to_process, failed_images):
    total_files = (len(image_names_to_process) + SAVE_INTERVAL - 1) // SAVE_INTERVAL
    for file_count in range(1, total_files + 1):
        save_partial_results(file_count, all_descriptions, image_names_to_process)
    
    if failed_images:
        failed_file = os.path.join(OUTPUT_DIR, 'failed_train_images.txt')
        with open(failed_file, 'w', encoding='utf-8') as f:
            for img_name in failed_images:
                f.write(f"{img_name}\n")
        print(f"Failed images saved to {failed_file}")
        print(f"Number of failed images: {len(failed_images)}")

def encode_image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Failed to read image {image_path}: {e}")
        return None

def generate_multiple_descriptions_batch(image_paths, img_names_batch):
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
    
    user_content = []
    for img_name, image_path in zip(img_names_batch, image_paths):
        base64_image = encode_image_to_base64(image_path)
        if base64_image:
            user_content.append({
                "type": "text",
                "text": f"Image {img_name}:"
            })
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
    
    if not user_content:
        return ""
    
    try:
        completion = client.chat.completions.create(
            model="doubao-seed-1-6-vision-250815",
            messages=[
                {"role": "system", "content": prompt1},
                {"role": "user", "content": user_content}
            ],
            temperature=0.8,
            max_tokens=5000,
        )
        
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"API call failed: {e}")
        return ""

def parse_response_text(response_text, expected_images_count):
    if not response_text:
        return None
    lines = [line.strip() for line in response_text.split('\n') if line.strip()]

    filtered_lines = []
    for line in lines:
        if any(ext in line for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']):
            continue
        elif re.match(r'^\d+\.\s*', line):
            line = re.sub(r'^\d+\.\s*', '', line)
            filtered_lines.append(line)
        else:
            filtered_lines.append(line)

    total_descriptions_needed = expected_images_count * 5
    if len(filtered_lines) < total_descriptions_needed:
        print(f"Warning: Only obtained {len(filtered_lines)} descriptions, {total_descriptions_needed} needed")
        return None
    

    descriptions_by_image = []
    for i in range(0, total_descriptions_needed, 5):
        image_descriptions = filtered_lines[i:i+5]
        if len(image_descriptions) != 5:
            print(f"Warning: Image {i//5} only has {len(image_descriptions)} descriptions")
            return None
        descriptions_by_image.append(image_descriptions)
    
    return descriptions_by_image


def main():

    signal.signal(signal.SIGINT, signal_handler)  
    signal.signal(signal.SIGTERM, signal_handler)  
    
    logger = Logger(LOG_FILE)
    sys.stdout = logger
    sys.stderr = logger
    
    try:
        print(f"{'='*60}")
        print("Program start time:", time.strftime('%Y-%m-%d %H:%M:%S'))
        print("Loading data...")
        image_names_to_process = load_data()
        total_images = len(image_names_to_process)
        print(f"Total {total_images} images to process")
        

        checkpoint = load_checkpoint()
        processed_count = checkpoint["processed_count"]
        file_count = checkpoint["file_count"]
        failed_images = checkpoint.get("failed_images", [])
        all_descriptions = checkpoint.get("all_descriptions", {})
        
        for img_name in image_names_to_process:
            if img_name not in all_descriptions:
                all_descriptions[img_name] = []
        
        print(f"Resuming from checkpoint: Processed {processed_count}/{total_images} images")
        print(f"Saved {file_count} files")
        print(f"Failed images count: {len(failed_images)}")
        print(f"Log file: {LOG_FILE}")
        print(f"Checkpoint file: {CHECKPOINT_FILE}")
        print(f"{'='*60}")
        
        pbar = tqdm(total=total_images, desc="Processing Progress", unit="img", 
                   mininterval=10.0, maxinterval=60.0)  # Reduce refresh frequency
        pbar.update(processed_count)
        
        start_time = time.time()
        last_save_time = time.time()
        
        for i in range(processed_count, total_images, BATCH_SIZE):
            current_time = time.time()
            
            if current_time - last_save_time > 300:  # Print status every 5 minutes
                elapsed_time = current_time - start_time
                processed_recently = processed_count - checkpoint.get("processed_count", 0)
                speed = processed_recently / max(elapsed_time, 1) * 3600  # img/hour
                print(f"\nStatus Report [{time.strftime('%H:%M:%S')}]:")
                print(f"  Processed: {processed_count}/{total_images} ({processed_count/total_images*100:.1f}%)")
                print(f"  Processing Speed: {speed:.1f} img/hr")
                print(f"  Elapsed Time: {elapsed_time/3600:.1f} hours")
                print(f"  Estimated Remaining: {(total_images-processed_count)/max(speed, 1):.1f} hours")
                last_save_time = current_time
            
            batch_img_names = image_names_to_process[i:i+BATCH_SIZE]
            batch_image_paths = []
            
            valid_img_names = []
            for img_name in batch_img_names:
                image_path = os.path.join(IMAGE_DIR, img_name)
                if os.path.exists(image_path):
                    batch_image_paths.append(image_path)
                    valid_img_names.append(img_name)
                else:
                    print(f"Error: Image does not exist: {image_path}")
                    print("Program paused, please check image files and rerun")
                    save_checkpoint(processed_count, file_count, failed_images, all_descriptions)
                    logger.close()
                    return
            
            if not batch_image_paths:
                continue
            
            batch_retry_count = 0
            batch_success = False
            
            while batch_retry_count < MAX_BATCH_RETRIES:
                api_retry_count = 0
                api_success = False
                
                while api_retry_count < MAX_API_RETRIES:
                    print(f"\n[{time.strftime('%H:%M:%S')}] Processing batch {i//BATCH_SIZE + 1}")
                    response_text = generate_multiple_descriptions_batch(batch_image_paths, valid_img_names)
                    
                    if response_text:
                        descriptions_by_image = parse_response_text(response_text, len(valid_img_names))
                        
                        if descriptions_by_image:
                            all_have_five = True
                            for idx, img_name in enumerate(valid_img_names):
                                if len(descriptions_by_image[idx]) != 5:
                                    print(f"Error: Image {img_name} only has {len(descriptions_by_image[idx])} descriptions")
                                    all_have_five = False
                                    break
                            
                            if all_have_five:
                                for idx, img_name in enumerate(valid_img_names):
                                    all_descriptions[img_name] = descriptions_by_image[idx]
                                print(f"✓ Successfully obtained descriptions for {len(valid_img_names)} images")
                                
                                api_success = True
                                batch_success = True
                                break
                    
                    api_retry_count += 1
                    if api_retry_count < MAX_API_RETRIES:
                        print(f"API call failed, retrying {api_retry_count}/{MAX_API_RETRIES}")
                        time.sleep(5)
                
                if batch_success:
                    processed_count += len(valid_img_names)
                    pbar.update(len(valid_img_names))
                    break
                else:
                    batch_retry_count += 1
                    if batch_retry_count < MAX_BATCH_RETRIES:
                        print(f"Batch processing failed, retrying {batch_retry_count}/{MAX_BATCH_RETRIES}")
                        time.sleep(10)
            
            if not batch_success:
                print(f"Critical Error: Batch {i//BATCH_SIZE + 1} processing failed, max retries reached")
                print(f"Failed images: {valid_img_names}")
                print("Program paused, please check API status and network connection, then rerun")
                
                for img_name in valid_img_names:
                    failed_images.append(img_name)
                
                save_checkpoint(processed_count, file_count, failed_images, all_descriptions)
                logger.close()
                return
            
            save_checkpoint(processed_count, file_count, failed_images, all_descriptions)
            
            if processed_count % SAVE_INTERVAL == 0 and processed_count > 0:
                file_count += 1
                save_partial_results(file_count, all_descriptions, image_names_to_process)
                save_checkpoint(processed_count, file_count, failed_images, all_descriptions)
            
            time.sleep(DELAY_BETWEEN_REQUESTS)
        
        pbar.close()
        print("Saving final results...")
        save_final_results(all_descriptions, image_names_to_process, failed_images)
        
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
        
        total_time = time.time() - start_time
        print(f"{'='*60}")
        print("Processing complete!")
        print(f"Completion time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total elapsed time: {total_time/3600:.2f} hours")
        print(f"Successfully processed: {total_images - len(failed_images)} images")
        print(f"Processing failed: {len(failed_images)} images")
        print(f"Generated files: {file_count} partial files")
        print(f"Log file: {LOG_FILE}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"Program exception occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.close()

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pid_file = os.path.join(OUTPUT_DIR, 'processing.pid')
    if os.path.exists(pid_file):
        with open(pid_file, 'r') as f:
            old_pid = f.read().strip()
        try:
            os.kill(int(old_pid), 0)
            print(f"Warning: Process {old_pid} is already running!")
            print("To start a new process, please delete: " + pid_file)
            sys.exit(1)
        except OSError:
            os.remove(pid_file)
    
    with open(pid_file, 'w') as f:
        f.write(str(os.getpid()))
    
    try:
        main()
    finally:
        if os.path.exists(pid_file):
            os.remove(pid_file)
