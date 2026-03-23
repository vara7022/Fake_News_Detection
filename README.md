# Fake News Detection using BERT — Mini Project

**Author:** Deekshith  
**Tool:** Python (Google Colab, T4 GPU)  
**Model:** DistilBERT fine-tuned for Sequence Classification  
**Dataset:** 24,353+ news articles (GonzaloA/fake_news via HuggingFace)

---

## Problem Statement

Build a robust, end-to-end fake news detection system that classifies news articles as **Real (0)** or **Fake (1)** using a pre-trained transformer model. The project covers the full ML lifecycle — from data analysis and preprocessing, through model fine-tuning and evaluation, to error analysis, class-imbalance handling, and live Gradio deployment.

---

## Approach

The project is structured as a 9-day development plan:

| Day | Task |
|---|---|
| Day 1–2 | Dataset loading, EDA, class distribution analysis, text length analysis, data cleaning |
| Day 3 | Tokenization with DistilBERT tokenizer, PyTorch Dataset/DataLoader pipeline |
| Day 4–5 | Model fine-tuning with manual training loop, early stopping, best checkpoint saving |
| Day 6 | Full evaluation — classification report, confusion matrix (raw + normalised), ROC-AUC |
| Day 7 | Error analysis — confidence-based FP/FN breakdown, high-confidence mistake inspection |
| Day 8 | Model improvement — weighted cross-entropy loss for class imbalance correction |
| Day 9 | Gradio deployment — live inference web app with confidence scores |

**Dataset Loading Strategy (Priority Order):**
1. Local CSV file (supports `WELFake_Dataset.csv`, `fake_news.csv`, `train.csv`, etc.)
2. HuggingFace Hub (`GonzaloA/fake_news` — 24K+ samples)
3. Fallback demo dataset (30 labelled samples for testing the pipeline)

**Data Cleaning:** URLs, email addresses, and excess whitespace are stripped. Samples shorter than 10 characters are dropped.

**Splits:** Stratified 70% / 15% / 15% train-validation-test split with `random_seed=42`.

---

## Model Used

**DistilBERT** (`distilbert-base-uncased`)  
`AutoModelForSequenceClassification` with `num_labels=2`

A manually written training loop (not the HuggingFace Trainer API) is used, giving full control over gradient clipping, custom loss functions, and early stopping logic.

| Hyperparameter | Value |
|---|---|
| Max sequence length | 512 tokens |
| Batch size | 16 |
| Epochs (max) | 5 |
| Learning rate | 2e-5 |
| LR scheduler | Linear warmup (10% of total steps) |
| Weight decay | 0.01 |
| Gradient clipping | 1.0 |
| Early stopping patience | 3 epochs |
| Optimizer | AdamW |

**Note on DistilBERT:** `token_type_ids` are explicitly excluded from all forward passes, as DistilBERT does not use segment embeddings.

---

## Metrics

Evaluation is performed on the held-out test set using the best checkpoint (saved whenever validation accuracy improves):

| Metric | Description |
|---|---|
| **Accuracy** | Overall fraction of correct predictions |
| **Precision** | Per-class: fraction of predicted positives that are correct |
| **Recall** | Per-class: fraction of actual positives correctly identified |
| **F1-Score** | Harmonic mean of precision and recall (weighted and per-class) |
| **ROC-AUC** | Area under the ROC curve — measures ranking quality |
| **Confusion Matrix** | Raw counts and normalised per-class accuracy |

**Expected results on the full 24K dataset:**

| Metric | Expected Value |
|---|---|
| Accuracy | 95%+ |
| F1-Score | 0.95+ |

**Error Analysis (Day 7):**
Misclassified samples are broken into False Positives (Real predicted as Fake) and False Negatives (Fake predicted as Real), with top high-confidence errors printed for manual inspection. A confidence distribution histogram separates correct vs. incorrect predictions.

**Weighted Loss Comparison (Day 8):**
A second model is trained with `nn.CrossEntropyLoss(weight=class_weights)` using `sklearn`'s `compute_class_weight('balanced')`. Validation accuracy is compared against the baseline to measure the benefit of imbalance correction.

---

## Improvements

- **Weighted cross-entropy loss** — `compute_class_weight('balanced')` computes per-class weights automatically and applies them to the loss function, improving recall on the minority class.
- **Early stopping with best checkpoint saving** — The model is saved only when validation accuracy improves, and training halts when patience is exhausted — preventing wasted compute and overfitting.
- **Gradient clipping** — `clip_grad_norm_(1.0)` prevents exploding gradients during fine-tuning.
- **Linear warmup scheduler** — LR is ramped up for the first 10% of steps before decaying linearly, which stabilises early training.
- **Flexible dataset loading** — The smart loader tries local files first, then HuggingFace, then a demo fallback — making the notebook portable across environments.

**Further possible improvements:**
- Try `bert-base-uncased` or `roberta-base` for potentially higher accuracy at the cost of more compute.
- Add label smoothing to reduce model overconfidence on ambiguous examples.
- Experiment with longer max_length (currently capped at 512, the DistilBERT limit).
- Deploy to HuggingFace Spaces for a permanent public demo instead of the temporary Colab `share=True` link.
- Add SHAP or attention visualisation for explainability.

---

## Key Learnings

- **Manual training loop vs. Trainer API** — Writing the training loop manually gives finer control (custom loss, per-step logic) at the cost of more boilerplate. The HuggingFace Trainer abstracts this away but is less flexible.
- **DistilBERT has no `token_type_ids`** — Passing `token_type_ids` to DistilBERT raises an error; all forward passes must explicitly exclude them. This is a common bug when adapting BERT code to DistilBERT.
- **Best checkpoint ≠ final checkpoint** — Saving the model only when validation accuracy improves, and loading it at inference time, is critical. Using the final epoch's weights can be worse if overfitting occurred.
- **Class imbalance matters** — Even moderate imbalance can bias the model toward the majority class. Weighted loss directly addresses this without needing data augmentation.
- **Confidence-based error analysis is more useful than raw counts** — High-confidence wrong predictions reveal where the model is fundamentally confused, not just uncertain.
- **Tokenization padding strategy** — Fixed `padding='max_length'` (used here) is simpler to implement but less efficient than dynamic padding. For production, `DataCollatorWithPadding` is preferred.

---

## Project Structure

```
fake_news_detection_BERT_MINI_PROJECT.ipynb   # Main notebook (9-day plan)
README.md                                      # This file
```

## Dependencies

```
transformers
datasets
accelerate
gradio
scikit-learn
seaborn
matplotlib
pandas
torch
```

## Quick Start

```
1. Runtime → Change runtime type → T4 GPU
2. Run all cells in order
3. Dataset loads automatically from HuggingFace if no local CSV is present
4. After training, the Gradio app launches with share=True for a public link
```
