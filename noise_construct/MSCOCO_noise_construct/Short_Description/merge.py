import os
import glob
import re
from tqdm import tqdm

def merge_noisy_files(input_dir, output_file, start_num=1, end_num=None):
    """
    Merges noisy_split_file_*.txt files sequentially into a single large file.
    
    Args:
        input_dir: Input file directory
        output_file: Output file path
        start_num: Start file number
        end_num: End file number (None for auto-detection)
    """
    
    print("Starting to merge noisy text files")
    print(f"Input directory: {input_dir}")
    print(f"Output file: {output_file}")
    
    if end_num is None:
        existing_files = glob.glob(os.path.join(input_dir, "noisy_split_file_*.txt"))
        if existing_files:
            file_numbers = []
            for f in existing_files:
                try:
                    num = int(re.findall(r'\d+', os.path.basename(f))[0])
                    file_numbers.append(num)
                except (IndexError, ValueError):
                    continue
            if file_numbers:
                end_num = max(file_numbers)
                print(f"Automatically detected file range: {start_num:03d} to {end_num:03d}")
            else:
                print("Error: Cannot extract numbers from filenames")
                return
        else:
            print("Error: No noisy_split_file_*.txt files found")
            return
    
    file_paths = []
    missing_files = []
    
    for file_num in range(start_num, end_num + 1):
        filename = f"noisy_split_file_{file_num:03d}.txt"
        file_path = os.path.join(input_dir, filename)
        if os.path.exists(file_path):
            file_paths.append(file_path)
        else:
            missing_files.append(filename)
    
    if not file_paths:
        print("Error: No valid files found")
        return
    
    print(f"Found {len(file_paths)} files")
    if missing_files:
        print(f"Warning: Missing {len(missing_files)} files: {', '.join(missing_files[:5])}{'...' if len(missing_files) > 5 else ''}")
    
    print("\nFile processing order:")
    for i, file_path in enumerate(file_paths[:10]):
        print(f"  {i+1:2d}. {os.path.basename(file_path)}")
    if len(file_paths) > 10:
        print(f"  ... and {len(file_paths) - 10} more files")
    
    print("\nCounting total lines...")
    total_lines = 0
    file_info = []
    
    for file_path in tqdm(file_paths, desc="Counting lines"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                line_count = len(lines)
                total_lines += line_count
                file_info.append({
                    'path': file_path,
                    'lines': line_count,
                    'content': lines
                })
        except Exception as e:
            print(f"Error reading file {os.path.basename(file_path)}: {e}")
            file_info.append({
                'path': file_path,
                'lines': 0,
                'content': []
            })
    
    print(f"Total lines: {total_lines:,}")
    
    print("\nStarting file merge...")
    successful_files = 0
    failed_files = 0
    
    try:
        with open(output_file, 'w', encoding='utf-8') as out_f:
            with tqdm(total=total_lines, desc="Merging progress", unit="line", ncols=80) as pbar:
                for info in file_info:
                    filename = os.path.basename(info['path'])
                    
                    if info['lines'] > 0:
                        try:
                            for line in info['content']:
                                out_f.write(line)
                            
                            successful_files += 1
                            pbar.update(info['lines'])
                            pbar.set_description(f"Merging: {filename}")
                            
                        except Exception as e:
                            print(f"Error writing file {filename}: {e}")
                            failed_files += 1
                            pbar.update(info['lines'])
                    else:
                        print(f"Skipping empty file: {filename}")
                        failed_files += 1
                        pbar.update(0)
    
    except Exception as e:
        print(f"Failed to create output file: {e}")
        return
    
    print("\nVerifying output file...")
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            output_lines = sum(1 for _ in f)
        print(f"Output file lines: {output_lines:,}")
        
        if output_lines == total_lines:
            print("File line count verification passed!")
        else:
            print(f"Line count mismatch: Expected {total_lines}, Actual {output_lines}")
            
    except Exception as e:
        print(f"Failed to verify output file: {e}")
    
    print("\nMerge completion statistics:")
    print(f"Successfully merged: {successful_files} files")
    print(f"Failed/Skipped: {failed_files} files")
    print(f"Total files: {len(file_paths)}")
    print(f"Total text lines: {total_lines:,}")
    print(f"Output file: {output_file}")

def find_missing_files(input_dir, start_num=1, end_num=None):
    if end_num is None:
        existing_files = glob.glob(os.path.join(input_dir, "noisy_split_file_*.txt"))
        if existing_files:
            file_numbers = [int(re.findall(r'\d+', os.path.basename(f))[0]) for f in existing_files]
            end_num = max(file_numbers)
        else:
            print("No files found")
            return
    
    missing_files = []
    for file_num in range(start_num, end_num + 1):
        filename = f"noisy_split_file_{file_num:03d}.txt"
        file_path = os.path.join(input_dir, filename)
        if not os.path.exists(file_path):
            missing_files.append(filename)
    
    if missing_files:
        print(f"Missing {len(missing_files)} files:")
        for i, filename in enumerate(missing_files):
            print(f"  {i+1:2d}. {filename}")
    else:
        print("All files exist")


if __name__ == "__main__":
    # Configuration parameters
    input_directory = "/path/noise_construct/MSCOCO_noise_cinstruct/incomplete_description/noise"
    output_file_path = "/path/noise_construct/MSCOCO_noise_cinstruct/incomplete_description/1.0_noise_train_caps_preliminary.txt"
    
    merge_noisy_files(
        input_dir=input_directory,
        output_file=output_file_path,
        start_num=1,
        end_num=None  
    )
