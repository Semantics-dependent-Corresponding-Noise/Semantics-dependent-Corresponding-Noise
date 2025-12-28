from pathlib import Path

def split_file_with_path(input_file, output_path, lines_per_file=1000):
    """
    Args:
        input_file: Path to the input file
        output_path: Output path (can be a directory or full file path)
        lines_per_file: Number of lines per file
    """
    
    try:
        input_path = Path(input_file)
        if not input_path.exists():
            print(f"Error: Input file '{input_file}' does not exist")
            return
        
        output_path = Path(output_path)
        
        if output_path.suffix == '':
            output_path.mkdir(parents=True, exist_ok=True)
            output_dir = output_path
            output_prefix = "split_file"
        else:
            output_dir = output_path.parent
            output_prefix = output_path.stem
            output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Input file: {input_path}")
        print(f"Output directory: {output_dir}")
        print(f"File prefix: {output_prefix}")
        
        file_count = 0
        line_count = 0
        
        with open(input_path, 'r', encoding='utf-8') as f:
            while True:
                file_count += 1
                output_file = output_dir / f"{output_prefix}_{file_count:03d}.txt"
                
                with open(output_file, 'w', encoding='utf-8') as out_f:
                    for _ in range(lines_per_file):
                        line = f.readline()
                        if not line:
                            break
                        out_f.write(line)
                        line_count += 1
                
                if output_file.stat().st_size == 0:
                    output_file.unlink()
                    file_count -= 1
                    break

                print(f"Created: {output_file}")
                
                if not line:
                    break
        
        print("\nSplitting complete!")
        print(f"Number of files generated: {file_count}")
        print(f"Total lines: {line_count}")
        print(f"Saved to: {output_dir.absolute()}")
        
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    split_file_with_path("/path/dataset/incomplete_description_noise_MSCOCO/annotations/scan_split/0_noise_train_caps.txt", "/path/noise_construct/MSCOCO_noise_cinstruct/incomplete_description/original", 1000)
    
