import os
import glob
import asyncio
import aiofiles
from collections import defaultdict

async def merge_description_files(input_dir, output_file):
    """
    Merges all split description files into a single file in numerical order.
    
    Args:
        input_dir: Directory containing the split files
        output_file: Path for the merged output file
    """
    
    pattern = os.path.join(input_dir, "train_caps_5_per_image_part*.txt")
    files = glob.glob(pattern)
    
    if not files:
        print(f"[Error] No matching files found in {input_dir}")
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
    
    print("\n" + "="*60)
    print("Starting file merge...")
    
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
        
        print("\n" + "="*60)
        print("Merge complete!")
        print(f"Statistics:")
        print(f"  Files merged: {total_files}")
        print(f"  Total images: {total_images:,}")
        print(f"  Total lines: {total_lines:,}")
        print(f"  Output file: {output_file}")
        
        # Verify output file
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"  File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
        
        return True
        
    except Exception as e:
        print(f"\n[Error] Merge failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def sync_merge_description_files(input_dir, output_file):

    pattern = os.path.join(input_dir, "train_caps_5_per_image_part*.txt")
    files = glob.glob(pattern)
    
    if not files:
        print(f"[Error] No matching files found in {input_dir}")
        return False

    files.sort(key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x))) or 0))
    
    total_files = len(files)
    
    print("="*60)
    print(f"Found {total_files} files")
    print("="*60)
    print("Starting merge...")
    
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
        
        print("\n" + "="*60)
        print("Merge complete!")
        print(f"Output file: {output_file}")

        if os.path.exists(output_file):
            total_lines = 0
            with open(output_file, 'r', encoding='utf-8') as f:
                for _ in f:
                    total_lines += 1
            
            print(f"Total lines: {total_lines:,}")
            print(f"Image count: {total_lines // 5:,}")
            print(f"File size: {os.path.getsize(output_file):,} bytes")
        
        return True
        
    except Exception as e:
        print(f"\n[Error] Merge failed: {e}")
        return False

if __name__ == "__main__":
    # Input directory (directory containing split files)
    INPUT_DIR = '/path/noise_construct/f30k_noise_construct/core_missing/train_flickr'
    
    # Output file (merged complete file)
    OUTPUT_FILE = '/path/dataset/core_missing_Error_noise_f30k/annotations/scan_split/1.0_noise_train_caps.txt'
    
    # Whether to use async mode (Recommended True for faster speed)
    USE_ASYNC = True
    
    print("Starting description file merge...")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Mode: {'Async' if USE_ASYNC else 'Sync'}")
    print("="*60)
    
    if USE_ASYNC:
        asyncio.run(merge_description_files(INPUT_DIR, OUTPUT_FILE))
    else:
        sync_merge_description_files(INPUT_DIR, OUTPUT_FILE)
    
    print("\nProgram finished")
