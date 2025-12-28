# Core Missing Noise File Generation Workflow

## 1. Project Structure

| File Name | Description |
| :--- | :--- |
| `yibu_batch_core_missing_noise_doubao_construct.py` | Training noise generation (100% error). |
| `yibu_batch_core_missing_test_noise_doubao_construct.py` | Test noise generation. |
| `merge_files.py` | Merges split files. |
| `core_missing_noise.py` | Generates partial error noise. |
| `core_missing_img5txt_noise.py` | Generates all error noise. |
| `./train_file/` | Training data output directory. |
| `./test_file/` | Test data output directory. |

---

## 2. Usage Steps

### Phase I: Generate Training Set Noise

**1. Generate Base Noise Files**
Run the command `python yibu_batch_core_missing_noise_doubao_construct.py` to generate base noise files containing **100% erroneous noise descriptions**, processing images sequentially in **batches of 1000**. The output files, named `train_caps_5_per_image_part{file_number}.txt`, will be saved to the `./train_file/` directory.

**2. Merge Files**
Run the command `python merge_files.py` to merge the split part files generated in the previous step into a single complete training noise file.

### Phase II: Generate Test Set Noise

**Generate Test Noise**
Run the command `python yibu_batch_core_missing_test_noise_doubao_construct.py` to directly generate a complete test set noise file.

### Phase III: Generate Noise with Specified Ratio

**1. Partial Error (Mixed)**
Run the command `python core_missing_noise.py` to generate dataset variations where **not all descriptions** for a given image are erroneous (standard setting: 1 image, 5 captions, with partial errors).

**2. All Error (Total)**
Run the command `python core_missing_img5txt_noise.py` to generate dataset variations where **all descriptions** for a given image are erroneous (setting: 1 image, 5 captions, 100% errors).

---

## 3. Configuration & Notes

### Configuration Details
*   **Input Data:** Ensure source image files are placed in the required directory structure.
*   **Output Directory:** Intermediate training files are automatically saved to `./train_file/`.
*   **Naming Convention:** Split files follow the format `train_caps_5_per_image_part{number}.txt`.

### Important Notes
1.  **Execution Order:** Please strictly follow the sequence: **Generate → Merge → Filter**.
2.  **Foundation:** Generating the 100% error noise (Phase I) is the necessary foundation for all subsequent operations.
3.  **Storage:** Ensure sufficient disk space is available for the generated text files.
4.  **Environment:** All scripts should be executed from the **project root directory**.
