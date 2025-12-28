# Incomplete Description Noise Generation Workflow

## 1. Project Structure

| File Name | Description |
| :--- | :--- |
| `incomplete_description_noise_construct.py` | Generates training noise (100% error). |
| `incomplete_description_test_noise_construct.py` | Generates test set noise. |
| `incomplete_description_noise.py` | Generates partial error noise (Mixed). |
| `incomplete_description_img5txt_error_noise.py` | Generates all-error noise (Total). |

---

## 2. Usage Steps

### Phase I: Generate Training Set Noise

**Generate Base Noise Files**
Run the command `python incomplete_description_noise_construct.py` to generate **100% erroneous noise descriptions** for the training set. This execution generates the foundational data file where every description contains incomplete description errors, which is necessary for creating subsequent datasets with specific noise ratios.

### Phase II: Generate Test Set Noise

**Generate Test Noise**
Run the command `python incomplete_description_test_noise_construct.py`. This script directly generates the complete noise file for the test set without requiring intermediate steps.

### Phase III: Generate Noise with Specified Ratio

After obtaining the complete base noise files from Phase I, use the following scripts to generate datasets with specific error ratios (assuming a standard setting where 1 image corresponds to 5 descriptions):

**1. Partial Error (Mixed)**
Run the command `python incomplete_description_noise.py` to generate a dataset where **not all** 5 descriptions for a given image are erroneous. This results in a mixed set of correct and noisy captions for specific images.

**2. All Error (Total)**
Run the command `python incomplete_description_img5txt_error_noise.py` to generate a dataset where **all 5** descriptions for a given image are erroneous (100% error rate for the selected images).

---

