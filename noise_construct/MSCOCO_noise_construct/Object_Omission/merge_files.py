import os
import glob
import asyncio
import aiofiles
from collections import defaultdict

async def merge_description_files(input_dir, output_file):
    """
    Merge all partitioned configuration files into a single file in numerical order.
    
    Parameters:
        input_dir: Directory containing split files
        output_file: Path of the merged output file
    """
    
    pattern = os.path.join(input_dir, "train_caps_5_per_image_part*.txt")
    files = glob.glob(pattern)
    
    if not files:
        print(f"Error: No matching files found in {input_dir}")
        return False
    
    def extract_number(filename):
        try:
            basename = os.path.basename(filename)
            number_part = ''.join(filter(str.isdigit, basename))
            return int(number_part) if number_part else 0
        except:
            return 0
    
    files.sort(key=extract_number)
    
    total_files = len(files)
    total_lines = 0
    total_images = 0


    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    
    try:
        async with aiofiles.open(output_file, 'w', encoding='utf-8') as outfile:
            for i, file_path in enumerate(files, 1):
                filename = os.path.basename(file_path)
                print(f"\rProcessing ({i}/{total_files}): {filename}", end="", flush=True)

                async with aiofiles.open(file_path, 'r', encoding='utf-8') as infile:
                    content = await infile.read()
                    lines = content.splitlines()
                    
                    if lines:
                        await outfile.write(content)
                        if content and not content.endswith('\n'):
                            await outfile.write('\n')
                        
                        total_lines += len(lines)
                        total_images += len(lines) // 5  
        
        return True
        
    except Exception as e:
        print(f"\nMerge failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def sync_merge_description_files(input_dir, output_file):
    pattern = os.path.join(input_dir, "train_caps_5_per_image_part*.txt")
    files = glob.glob(pattern)
    
    if not files:
        print(f"Error: No matching files found in {input_dir}")
        return False
    
    # Sorting
    files.sort(key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x))) or 0))
    total_files = len(files)
    
    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for i, file_path in enumerate(files, 1):
                filename = os.path.basename(file_path)
                print(f"\rProcessing ({i}/{total_files}): {filename}", end="", flush=True)
                
                with open(file_path, 'r', encoding='utf-8') as infile:
                    content = infile.read()
                    outfile.write(content)
                    if content and not content.endswith('\n'):
                        outfile.write('\n')   

        if os.path.exists(output_file):
            total_lines = 0
            with open(output_file, 'r', encoding='utf-8') as f:
                for _ in f:
                    total_lines += 1
            
            print(f"Total number of lines: {total_lines:,}")
            print(f"Number of images: {total_lines // 5:,}")
            print(f"File size: {os.path.getsize(output_file):,} byte")
        
        return True
        
    except Exception as e:
        print(f"\nMerge failed: {e}")
        return False

if __name__ == "__main__":
    # Input directory (containing the directory with split files)
    INPUT_DIR = '/path/noise_construct/MSCOCO_noise_cinstruct/core_missing/train_file'
    
    # Output file (merged complete file)
    OUTPUT_FILE = '/path/dataset/core_missing_Error_noise_MSCOCO/annotations/scan_split/1.0_noise_train_caps.txt'
    
    # Whether to use asynchronous mode (recommended: True, for faster performance)
    USE_ASYNC = True

    if USE_ASYNC:
        asyncio.run(merge_description_files(INPUT_DIR, OUTPUT_FILE))
    else:
        sync_merge_description_files(INPUT_DIR, OUTPUT_FILE)
