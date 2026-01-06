# Hidden-Secrets-Detection-using-Hybrid-Multi-Task-Transformer

This project explores a **hybrid multi-task learning approach** for detecting hard-coded secrets in source code. By jointly training on **classification**, **span localization**, and **entropy regression**, the model moves beyond simple pattern matching to a deeper semantic understanding of code context. 


## Project Overview

Traditional tools like Gitleaks and TruffleHog are powerful but brittle; they struggle when secrets are obfuscated or placed in unusual contexts. This project leverages a **CodeBERT** encoder to reason over the semantic usage of strings, maintaining high recall even under adversarial conditions. 

I have uploaded most of the dataset (generated) and code files that I used during this project.

### Key Highlights

* **Multi-Task Architecture**: Jointly predicts secret presence, exact location (span), and Shannon entropy. 


* **Adversarial Robustness**: Specifically designed to handle "unseen" obfuscations such as Base64 encoding and string concatenation. 


* **Bias Correction**: Implements prefix padding and structural wrappers to eliminate positional bias in span detection. 


## Model Architecture

The core of the project is a **shared CodeBERT encoder** branching into three specialized task heads. 

### 1. The Task Heads

* **Classification Head**: Performs binary classification (Secret vs. No Secret). 


* **Span Heads**: Predicts token-level start and end positions for localization. 


* **Entropy Head**: A regression head that estimates the statistical randomness of the identified string. 

<img src="/misc/architecture_diagram.png" alt="Architecture Diagram" width="100%" />

### 2. Multi-Task Loss Strategy

To balance these objectives, we use a weighted loss function:

Total Loss = c1 * BCE + c2 * MSE + c3 * SpanLoss

Where:
- BCE = Binary Cross-Entropy (classification)
- MSE = Mean Squared Error (entropy regression)
- SpanLoss = masked start/end span loss
- c1, c2, c3 = loss weights

We employed a two-stage training strategy:

* **Warm-up (2 Epochs)**: Encoder is frozen to stabilize task heads and prevent "catastrophic forgetting." 


* **Fine-tuning (5 Epochs)**: Full model is unfrozen with increased weights for entropy () and span () to force precision in localization. 



## Dataset Construction

The dataset was meticulously curated to prevent the model from learning "shortcuts" like "high entropy always equals a secret." 

| Category | Description | Significance |
|---------|-------------|--------------|
| **Real Positives** | Leaks extracted from GitHub repositories and filtered for quality. | Provides real-world ground truth. |
| **Synthetic Positives** | Programmatically injected secrets using encoding, concatenation, and indirection. | Teaches robustness against obfuscation. |
| **Hard Negatives** | High-entropy non-secrets such as UUIDs, hashes, and request IDs. | Prevents false positives based on randomness alone. |
| **Contextual Negatives** | Benign code contextually similar to positives (e.g., non-secret identifiers). | Forces semantic differentiation instead of pattern matching. |

## Evaluation Results

The model was benchmarked against a baseline classifier and standard regex-based tools using a dedicated adversarial test set. 

| Model | Precision | Recall | F1 Score |
|------|-----------|--------|----------|
| **Hybrid Multi-Task Model** | 0.812 | **0.839** | **0.825** |
| Baseline Classifier | 0.845 | 0.681 | 0.755 |
| Gitleaks (Regex-based) | **1.000** | 0.538 | 0.700 |

| Task | Metric | Value |
|-----|--------|-------|
| Span Localization | Exact Match Accuracy (%) | **90.63** |
| Span Samples Evaluated | Count | **288** |
| Entropy Regression | Mean Absolute Error (MAE) | **0.714** |

* **Observation**: While Gitleaks has perfect precision, it misses nearly 50% of secrets under adversarial pressure. 

* **Conclusion**: Our hybrid model achieves a **23% improvement in recall** over Gitleaks while maintaining a 90.6% exact match accuracy for span localization. 


## Project Components and directories

* **`CLI_scanner/scan_secrets.py`**: The CLI implementation that utilizes the model's span predictions for localizing secrets in real files. 
* **`adversarial_testing/`**: Contains the dataset generations and evaluation scripts, dataset and results (json) innvolved in adversarial testing
* **`baseline_model/model.py`**: The baseline CodeBERT model
* **`data/`**: Contains all the data used and generated during the project
* **`hybrid_model_architecture/hybrid_dataset.py`**: Preprocessing and Tokenisation of the data
* **`hybrid_model_architecture/hybrid_model.py`**: Implementation of the CodeBERT-based multi-task architecture.
* **`hybrid_model_architecture/train_hybrid.py`**: The two-stage training pipeline (warm-up + fine-tuning).
* **`scripts/`**: Contains all the code files used to generate and refine different classes of data
* **`span_fixing/`**: The script used to fix the span start bias in original dataset



### CLI Scanner

The CLI scanner allows you to run the Hybrid Multi-Task model against local files or entire repositories.

Basic command : (make sure to run it from the parent project directory)\
**`python scan_secrets.py scan <path_to_scan> --model <path_to_checkpoint.pt>`**

#### CLI Options

| Option | Description | Default |
|------|-------------|---------|
| `path` | **Required.** File or directory to scan for secrets. | — |
| `--model` | **Required.** Path to the trained `best_model.pt` checkpoint. | — |
| `--threshold` | Confidence score (0.0–1.0) above which a finding is reported. | `0.7` |
| `--format` | Output style: `text` for human-readable console output, `json` for machine processing. | `text` |
| `--output` | Optional filename to save the results instead of printing to console. | Console Output |

#### Sample output :

<img src="/misc/sample_output.png" alt="Sample Output" width="100%" />


## Minor Implementation Details

* **Span Masking**: Span loss is only computed for samples with valid span annotations to avoid label noise. 
* **Rule Mapping**: Broad vendor categories (e.g., Parsehub, Yelp) were mapped to a high-entropy `generic-api-key` rule to ensure dataset coverage.
* **Prefix Padding**: Used to correct the "Start = 0" span bias found in the initial data. 


Self Project | Deep Learning for Security Research

By Rishav Kumar
