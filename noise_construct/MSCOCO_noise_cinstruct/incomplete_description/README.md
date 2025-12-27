# Incomplete Description Noise Generation Workflow

## 1. Project Structure

| File Name | Description |
| :--- | :--- |
| `split_file.py` | Splits the original training file into smaller parts (1000 lines each) to reduce generation errors. |
| `merge.py` | Merges the split noise files into a preliminary total file. |
| `incomplete_description_noise_construct.py` | Generates noise for each split file sequentially. |
| `incomplete_desrciption_noise_complete_construct.py` | Compares the noise file with the original, re-modifies identical lines to ensure 100% noise, and saves the final file. |
| `incomplete_description_test_noise_construct.py` | Generates noise for the test set directly. |
| `incomplete_description_noise.py` | Generates partial error noise (some descriptions contain errors). |
| `incomplete_description_img5txt_error_noise.py` | Generates all-error noise (all descriptions contain errors). |
| `./original/` | Directory for storing split source files. |
| `./noise/` | Directory for storing generated split noise files. |

---

## 2. Usage Steps

### Phase I: Generate Training Set Noise

**1. Split Original File**
Run the command `python split_file.py`.
- **Action:** Splits the original file `0_noise_train_caps.txt` into multiple small files (1000 lines per file).
- **Output Location:** Saved to the `./original/` directory.
- **Purpose:** To reduce the probability of errors during large-scale API calling.

**2. Generate Batch Noise Files**
Run the command `python incomplete_description_noise_construct.py`.
- **Action:** Sequentially generates the corresponding noise file for each small file in the `original` directory.
- **Output Location:** Saved to the `./noise/` directory.

**3. Merge Files**
Run the command `python merge.py`.
- **Action:** Merges the small noise files from the `noise` directory in order into a single preliminary file.
- **Output File:** `1.0_noise_train_caps_preliminary.txt`.

**4. Refine and Finalize (Ensure 100% Noise)**
Run the command `python incomplete_desrciption_noise_complete_construct.py`.
- **Action:** Compares `1.0_noise_train_caps_preliminary.txt` with the original `0_noise_train_caps.txt`. If identical lines are found (indicating failed modification), it re-calls the API to modify them.
- **Output File:** `1.0_noise_train_caps.txt` (The final 100% noise file).

### Phase II: Generate Test Set Noise

**Generate Test Noise**
Run the command `python incomplete_description_test_noise_construct.py`.
- **Action:** Directly generates the noise file for the test set (no splitting is required due to the smaller data size).

### Phase III: Generate Noise with Specified Ratio

**1. Partial Error (Mixed)**
Run the command `python incomplete_description_noise.py`.
- **Description:** Generates dataset variations where **not all** 5 descriptions for a given image are erroneous (setting: 1 image, 5 captions, with partial errors).

**2. All Error (Total)**
Run the command `python incomplete_description_img5txt_error_noise.py`.
- **Description:** Generates dataset variations where **all 5** descriptions for a given image are erroneous (setting: 1 image, 5 captions, 100% errors).

---

## 3. Configuration & Notes

### Configuration Details
*   **Directory Structure:** Ensure the `original` and `noise` directories exist or allow the scripts to create them.
*   **Input Data:** The process starts with `0_noise_train_caps.txt`.

### Important Notes
1.  **Execution Order:** Phase I must be executed strictly in the order: **Split → Generate → Merge → Refine**.
2.  **Quality Control:** Step 4 in Phase I is critical to ensure that the noise ratio is strictly 1.0 (100% difference from the original).
3.  **Test Set:** The test set does not require the split-merge-refine process; it is generated in one pass.
