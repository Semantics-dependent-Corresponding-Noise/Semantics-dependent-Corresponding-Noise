# Core Missing (f30k) Noise Generation Workflow

## 1. Project Structure

| File Name | Description |
| :--- | :--- |
| `f30k_object_omission_noise_doubao.py` | Generates training noise (100% error) in batches. |
| `f30k_object_omission_test_noise_doubao.py` | Generates test set noise. |
| `merge_files.py` | Merges the split batch files into a single complete file. |
| `f30k_object_omission_noise.py` | Generates partial error noise (Mixed). |
| `f30k_object_omission_img5txt_noise.py` | Generates all-error noise (Total). |
| `./train_flickr/` | Directory where training batch output files are saved. |
| `./test_flickr/` | Directory where test batch output files are saved. |

---

## 2. Usage Steps

### Phase I: Generate Training Set Noise

**Generate Base Noise Files**
Run the command `python f30k_object_omission_noise_doubao.py` to generate **100% erroneous noise descriptions**. The script processes images sequentially and outputs the data in batches of **1000 images**. The resulting split files follow the naming convention `train_caps_5_per_image_part{file_number}.txt` and are automatically saved to the `./train_flickr/` directory.

**Merge Files**
Run the command `python merge_files.py`. This script scans the `./train_flickr/` directory for the split part files generated in the previous step and merges them into a single, complete training noise file.

### Phase II: Generate Test Set Noise

**Generate Test Noise**
First, run the command `python f30k_object_omission_test_noise_doubao.py` to create the split files. Then, run the command `python merge_files.py` to combine them. **Note:** You must modify three variables within the `merge_files.py` script before running it for the test set: `pattern`, `INPUT_DIR`, and `OUTPUT_FILE`. Configure the script to scan the `./test_flickr/` directory for files matching the format `test_caps_5_per_image_part{file_number}.txt` and merge them into a single output file named `text_core.txt`.

### Phase III: Generate Noise with Specified Ratio

After obtaining the complete base noise files from Phase I and II, use the following scripts to generate datasets with specific error ratios (assuming a standard setting where 1 image corresponds to 5 descriptions):

**1. Partial Error (Mixed)**
Run the command `python f30k_object_omission_noise.py` to generate a dataset where **not all** 5 descriptions for a given image are erroneous. This creates a mixed dataset containing both correct and noisy captions for the same image.

**2. All Error (Total)**
Run the command `python f30k_object_omission_img5txt_noise.py` to generate a dataset where **all 5** descriptions for a given image are erroneous (100% error rate for the selected images).

---

## 3. Configuration & Notes

### Configuration Details
Ensure that the `./train_flickr/` directory exists or that the script has the necessary permissions to create it, as this is where the intermediate batch files are stored. The merge script relies on the specific filename pattern `train_caps_5_per_image_part{file_number}.txt` to correctly identify and combine the files.

### Important Notes
Please strictly follow the execution order: **Generate (Batch) → Merge → Ratio Generation**. Generating the 100% error base file is the mandatory foundation for all subsequent operations. Additionally, ensure sufficient disk space is available, as the batch generation process creates multiple intermediate text files in the `train_flickr` directory.
