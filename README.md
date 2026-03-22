# Explainable ML for Early Epileptic Seizure Detection using EEG

An explainable machine learning pipeline for early epileptic seizure detection using EEG brain signals.  
Research published and presented at **IEEE DELCON 2025**.  
📄 DOI: [10.1109/DELCON68055.2025.11400324](https://doi.org/10.1109/DELCON68055.2025.11400324)

---

## Overview

Epilepsy affects over 50 million people globally. Early warning systems — ones that flag a likely seizure minutes before onset — can dramatically improve patient safety. This project builds a feature-based ML pipeline that provides seizure warnings **4–15 minutes before onset** while remaining interpretable enough for clinical use.

The core challenge: seizures are rare (< 1% of EEG data), making this a highly imbalanced classification problem. The solution prioritizes both reliable detection and explainability via SHAP.

---

## Results

| Metric | Score |
|---|---|
| Average Accuracy | 89.74% |
| F1-Score (imbalanced test) | 74.39% |
| Early Warning Window | 4–15 minutes before onset |
| Evaluation Strategy | Stratified 10-fold cross-validation |

---

## Technical Approach

**Dataset:** CHB-MIT Scalp EEG Database (PhysioNet) — long-term EEG recordings from 23 pediatric patients with intractable epilepsy. Subset used: patients chb01–chb10.

**Pipeline:**
1. **Preprocessing** — segment raw EEG into 4-second windows with 2-second overlap
2. **Feature Engineering** — extract 322 handcrafted features across three domains:
   - Time-domain (mean, variance, skewness, Hjorth parameters)
   - Frequency-domain (band power: delta, theta, alpha, beta, gamma)
   - Non-linear (sample entropy, Hurst exponent)
3. **Dimensionality Reduction** — PCA reduces 322 features to 170
4. **Imbalance Handling** — SMOTE applied exclusively within training folds (prevents data leakage)
5. **Model** — LightGBM classifier
6. **Explainability** — SHAP analysis identifies top predictive features

---

## Repo Structure

```
├── eeg_preprocessing/    # Signal preprocessing
├── features.py           # Feature extraction (time, frequency, non-linear)
├── training.py           # LightGBM training with stratified CV + SMOTE
├── evaluation.py         # Metrics and cross-validation results
├── shap_summary.py       # SHAP explainability analysis
├── plots.py              # Visualizations
└── config.py             # Global parameters
```

---

## Setup

```bash
pip install -r requirements.txt
```

**Dataset:** Download the CHB-MIT Scalp EEG Database from [PhysioNet](https://physionet.org/content/chbmit/1.0.0/) and place it in `data/raw/`.

---

## Key Design Decisions

- **LightGBM over deep learning** — dataset size makes deep models prone to overfitting; LightGBM generalizes well and is faster to interpret
- **SMOTE inside folds only** — applying SMOTE before splitting leaks synthetic test samples into training, artificially inflating scores
- **SHAP for explainability** — clinical deployment requires transparency; SHAP provides feature-level explanations doctors can act on
- **F1 as primary metric** — accuracy is misleading under severe class imbalance

---

## Citation

```
Chetna et al., "Improving Healthcare Outcomes with Explainable Machine Learning 
for Early Epileptic Seizure Detection Using EEG Signals," 
IEEE DELCON 2025. DOI: 10.1109/DELCON68055.2025.11400324
```
