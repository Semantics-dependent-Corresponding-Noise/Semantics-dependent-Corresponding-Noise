# Core Missing (Doubao) Noise Generation Workflow

## 1. Project Structure

| File Name | Description |
| :--- | :--- |
| `batch_core_missing_noise_doubao_construct.py` | Generates training noise (100% error) in batches. |
| `batch_core_missing_test_noise_doubao_construct.py` | Generates test set noise. |
| `merge_files.py` | Merges the split batch files into a single complete file. |
| `core_missing_noise.py` | Generates partial error noise (Mixed). |
| `core_missing_img5txt_noise.py` | Generates all-error noise (Total). |
| `./train_flickr/` | Directory where batch output files are saved. |

---

## 2. Usage Steps

### Phase I: Generate Training Set Noise

**Generate Base Noise Files**
Run the command `python batch_core_missing_noise_doubao_construct.py` to generate **100% erroneous noise descriptions**. The script processes images sequentially and outputs the data in batches of **1000 images**. The resulting split files follow the naming convention `train_caps_5_per_image_part{file_number}.txt` and are automatically saved to the `./train_flickr/` directory.

**Merge Files**
Run the command `python merge_files.py`. This script scans the `./train_flickr/` directory for the split part files generated in the previous step and merges them into a single, complete training noise file.

### Phase II: Generate Test Set Noise

**Generate Test Noise**
Run the command `python batch_core_missing_test_noise_doubao_construct.py`. This script directly generates the complete noise file for the test set, following the same logic used for the training set generation but without the need for subsequent merging.

### Phase III: Generate Noise with Specified Ratio

After obtaining the complete base noise files from Phase I and II, use the following scripts to generate datasets with specific error ratios (assuming a standard setting where 1 image corresponds to 5 descriptions):

**1. Partial Error (Mixed)**
Run the command `python core_missing_noise.py` to generate a dataset where **not all** 5 descriptions for a given image are erroneous. This creates a mixed dataset containing both correct and noisy captions for the same image.

**2. All Error (Total)**
Run the command `python core_missing_img5txt_noise.py` to generate a dataset where **all 5** descriptions for a given image are erroneous (100% error rate for the selected images).

---

## 3. Configuration & Notes

### Configuration Details
Ensure that the `./train_flickr/` directory exists or that the script has the necessary permissions to create it, as this is where the intermediate batch files are stored. The merge script relies on the specific filename pattern `train_caps_5_per_image_part{file_number}.txt` to correctly identify and combine the files.

### Important Notes
Please strictly follow the execution order: **Generate (Batch) → Merge → Ratio Generation**. Generating the 100% error base file is the mandatory foundation for all subsequent operations. Additionally, ensure sufficient disk space is available, as the batch generation process creates multiple intermediate text files in the `train_flickr` directory.
