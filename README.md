# Hidden-Secrets-Detection-using-Hybrid-Multi-Task-Transformer

This project explores a **hybrid multi-task learning approach** for detecting hard-coded secrets in source code. By jointly training on **classification**, **span localization**, and **entropy regression**, the model moves beyond simple pattern matching to a deeper semantic understanding of code context. 

---

## Project Overview

Traditional tools like Gitleaks and TruffleHog are powerful but brittle; they struggle when secrets are obfuscated or placed in unusual contexts. This project leverages a **CodeBERT** encoder to reason over the semantic usage of strings, maintaining high recall even under adversarial conditions. 

### Key Highlights

* 
**Multi-Task Architecture**: Jointly predicts secret presence, exact location (span), and Shannon entropy. 


* 
**Adversarial Robustness**: Specifically designed to handle "unseen" obfuscations such as Base64 encoding and string concatenation. 


* 
**Bias Correction**: Implements prefix padding and structural wrappers to eliminate positional bias in span detection. 



---

## 🧠 Model Architecture

The core of the project is a **shared CodeBERT encoder** branching into three specialized task heads. 

### 1. The Task Heads

* 
**Classification Head**: Performs binary classification (Secret vs. No Secret). 


* 
**Span Heads**: Predicts token-level start and end positions for localization. 


* 
**Entropy Head**: A regression head that estimates the statistical randomness of the identified string. 



### 2. Multi-Task Loss Strategy

To balance these objectives, we use a weighted loss function:




We employed a two-stage training strategy:

* 
**Warm-up (2 Epochs)**: Encoder is frozen to stabilize task heads and prevent "catastrophic forgetting." 


* 
**Fine-tuning (5 Epochs)**: Full model is unfrozen with increased weights for entropy () and span () to force precision in localization. 



---

## 📊 Dataset Construction

The dataset was meticulously curated to prevent the model from learning "shortcuts" like "high entropy always equals a secret." 

| Category | Description | Significance |
| --- | --- | --- |
| **Real Positives** | Leaks extracted from GitHub and filtered for quality. 

 | Provides real-world ground truth. |
| **Synthetic Positives** | Programmatically injected secrets using encoding, concatenation, and indirection. 

 | Teaches robustness against obfuscation. |
| **Hard Negatives** | High-entropy non-secrets like UUIDs, Hashes, and Request IDs. 

 | Prevents false positives based on randomness alone. |
| **Contextual Negatives** | Benign code contextually similar to positives (e.g., non-secret identifiers). 

 | Forces semantic differentiation. |

---

## 🧪 Evaluation Results

The model was benchmarked against a baseline classifier and standard regex-based tools using a dedicated adversarial test set. 

### Adversarial Benchmarking Results

| Model | Precision | Recall | F1 Score |
| --- | --- | --- | --- |
| **Hybrid Multi-Task Model** | 0.812 | **0.839** | **0.825** |
| Baseline Classifier | 0.845 | 0.681 | 0.755 |
| Gitleaks (Regex) | **1.000** | 0.538 | 0.700 |

* 
**Observation**: While Gitleaks has perfect precision, it misses nearly 50% of secrets under adversarial pressure. 


* 
**Conclusion**: Our hybrid model achieves a **23% improvement in recall** over Gitleaks while maintaining a 90.6% exact match accuracy for span localization. 



---

## 🛠 Project Components

* **`hybrid_model.py`**: Implementation of the CodeBERT-based multi-task architecture.
* **`train_hybrid.py`**: The two-stage training pipeline (warm-up + fine-tuning).
* 
**`scan_secrets.py`**: A CLI implementation that utilizes the model's span predictions for localizing secrets in real files. 



### CLI Scanner Logic

The scanner utilizes a **sliding window approach** (overlapping chunks) to ensure secrets aren't split at buffer edges. It uses the model's span prediction as a **semantic anchor**, which is then refined to extract the precise credential.

---

## 📌 Minor Implementation Details

* 
**Span Masking**: Span loss is only computed for samples with valid span annotations to avoid label noise. 


* **Rule Mapping**: Broad vendor categories (e.g., Parsehub, Yelp) were mapped to a high-entropy `generic-api-key` rule to ensure dataset coverage.
* 
**Prefix Padding**: Used to correct the "Start = 0" span bias found in the initial data. 



**Author:** Rishav Kumar
*Deep Learning for Security Research*
