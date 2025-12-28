# K-Means Noise File Generation Workflow

## 1. Project Structure

| File Name | Description |
| :--- | :--- |
| `k_means_noise_construct.py` | Generates training noise (100% error). |
| `k_means_test_noise_construct.py` | Generates test set noise. |
| `k_means_noise.py` | Generates partial error noise. |
| `k_means_img5txt_error_noise.py` | Generates all error noise. |

---

## 2. Usage Steps

### Phase I: Generate Base Noise Files

**1. Generate Training Set Noise (100% Error)**
   Command: `python k_means_noise_construct.py`
   - **Action:** Generates 100% erroneous noise descriptions for the training set.
   - **Purpose:** This file serves as the necessary foundation for generating specific noise ratios later.

**2. Generate Test Set Noise**
   Command: `python k_means_test_noise_construct.py`
   - **Action:** Directly generates the complete noise file for the test set.

### Phase II: Generate Noise with Specified Ratio

After obtaining the complete base noise files from Phase I, use the following scripts based on your specific requirements (assuming 1 image corresponds to 5 descriptions):

- **Partial Error (Mixed)**
  Command: `python k_means_noise.py`
  - **Description:** Generates a dataset where **not all** 5 descriptions for a given image are erroneous (mixed correct and noisy captions).

- **All Error (Total)**
  Command: `python k_means_img5txt_error_noise.py`
  - **Description:** Generates a dataset where **all 5** descriptions for a given image are erroneous (100% error for selected images).
