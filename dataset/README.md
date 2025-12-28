# Dataset & Noise Annotations Repository

This directory contains the generated noise datasets used for training and testing. The data encompasses various types of textual noise (hallucinations, omissions, etc.) applied to both the **MSCOCO** and **Flickr30k (f30k)** datasets.

## 📂 Directory Structure Overview

The folders are organized by **Error Type**, **Noise Intensity (Condition)**, and **Source Dataset**.

```text
dataset/
├── Entity_Referential_Error_noise_5error_MSCOCO/    # Entity errors (All 5 captions noisy) - COCO
├── Entity_Referential_Error_noise_5error_f30k/      # Entity errors (All 5 captions noisy) - Flickr30k
├── Entity_Referential_Error_noise_MSCOCO/           # Entity errors (Partial/Mixed noise) - COCO
├── Entity_Referential_Error_noise_f30k/             # Entity errors (Partial/Mixed noise) - Flickr30k
├── High_level_Semantic_Confusion_5error_MSCOCO/     # Semantic errors (All 5 captions noisy) - COCO
├── High_level_Semantic_Confusion_5error_f30k/       # Semantic errors (All 5 captions noisy) - Flickr30k
├── High_level_Semantic_Confusion_MSCOCO/            # Semantic errors (Partial/Mixed noise) - COCO
├── High_level_Semantic_Confusion_f30k/              # Semantic errors (Partial/Mixed noise) - Flickr30k
├── Object_Omission_noise_5error_MSCOCO/             # Omission errors (All 5 captions noisy) - COCO
├── Object_Omission_noise_5error_f30k/               # Omission errors (All 5 captions noisy) - Flickr30k
├── Object_Omission_noise_MSCOCO/                    # Omission errors (Partial/Mixed noise) - COCO
├── Object_Omission_noise_f30k/                      # Omission errors (Partial/Mixed noise) - Flickr30k
├── Short_Description_noise_5error_MSCOCO/           # Incomplete errors (All 5 captions noisy) - COCO
├── Short_Description_noise_5error_f30k/             # Incomplete errors (All 5 captions noisy) - Flickr30k
├── Short_Description_noise_MSCOCO/                  # Incomplete errors (Partial/Mixed noise) - COCO
└── Short_Description_noise_f30k/                    # Incomplete errors (Partial/Mixed noise) - Flickr30k
```
## 🏷️ Naming Convention

Folders are named using the following pattern:

`{Error_Type}{Condition}{Dataset}/annotations`


### 1. Error Types

**Entity_Referential_Error**:  
Hallucination where entities in the image are replaced with incorrect objects (e.g., "cat" becomes "dog").

**High_level_Semantic_Confusion**:  
Complex errors where the scene context, relationships, or actions are described incorrectly, leading to semantic confusion.

**Object_Omission_noise**:  
Key objects present in the image are missing from the description.

**Short_Description_noise (Incomplete)**:  
Descriptions are severely truncated or lack necessary detail, providing only a very brief summary.

### 2. Conditions (Noise Intensity)

**`_5error` suffix**:  
**Total Error**: Indicates that all 5 captions corresponding to a single image have been modified to contain noise (100% noisy captions per image).

**No suffix (Standard)**:  
**Partial/Mixed Error**: Indicates a mixed dataset where valid noise ratios (e.g., 20%, 40%, 60%) are stored. Not all captions for a single image are necessarily noisy.

### 3. Datasets

**MSCOCO**: Microsoft COCO dataset.

**f30k**: Flickr30k dataset.

## 📝 Usage

- Each folder typically contains an `annotations/` subdirectory with text files named according to the split (train/test) and the specific noise ratio (e.g., `0.6_noise_train_caps.txt`, `1.0_noise_train_caps.txt`).
- **For robust training experiments**: Use the `_5error` folders to train on data where every caption is noisy.
- **For ratio sensitivity analysis**: Use the standard folders (without `_5error`) to access datasets with varying degrees of noise (0.2, 0.4, 0.6, etc.).
