Noise File Generation Workflow
Project Structure
复制
.
├── yibu_batch_core_missing_noise_doubao_construct.py      # Training noise generation (100% error)
├── yibu_batch_core_missing_test_noise_doubao_construct.py # Test noise generation
├── merge_files.py                                         # Merge split files
├── core_missing_noise.py                                  # Generate partial error noise
├── core_missing_img5txt_noise.py                          # Generate all error noise
└── train_file/                                            # Training data output directory
Usage Steps
I. Generate Training Set Noise
1. Generate Base Noise Files
bash
复制
python yibu_batch_core_missing_noise_doubao_construct.py
Generate 100% erroneous noise descriptions
Process in batches of 1000 images sequentially
Output: train_caps_5_per_image_part{file_number}.txt
Saved to: ./train_file/ directory
2. Merge Files
bash
复制
python merge_files.py
Merge split part files into a complete training noise file
II. Generate Test Set Noise
bash
复制
python yibu_batch_core_missing_test_noise_doubao_construct.py
Directly generate a complete test set noise file
III. Generate Noise with Specified Ratio
Choose based on requirements:
表格
复制
Scenario	Command	Description
Partial Error (1 img, 5 caps, some errors)	python core_missing_noise.py	Not all descriptions are erroneous
All Error (1 img, 5 caps, all errors)	python core_missing_img5txt_noise.py	All descriptions are erroneous
Configuration Notes
Input Data: Ensure image files are placed as required
Output Directory: Intermediate training files are automatically saved to ./train_file/
Naming Format: train_caps_5_per_image_part{number}.txt
Important Notes
Execution order: Generate → Merge → Filter
Generating 100% error noise is the foundation for subsequent operations
Ensure sufficient disk space
All scripts should be run from the project root directory
