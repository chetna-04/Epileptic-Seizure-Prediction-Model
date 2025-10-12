**Improving Healthcare Outcomes with Explainable Machine Learning for Early Epileptic Seizure Detection Using EEG Signals
Overview**
This repository contains the complete code and configuration for the explainable machine learning (XML) framework for early epileptic seizure detection using Electroencephalogram (EEG) signals.
The core contribution of this work is the development and evaluation of a feature-based Light Gradient Boosting Machine (LGBM) classifier, which provides early seizure warnings while maintaining high interpretability using SHapley Additive exPlanations (SHAP) under realistic, imbalanced testing conditions.

**Key Features**
1. Early Warning System: Provides seizure warnings between 4 and 15 minutes before onset.
2. Explainable AI (XAI): Implements SHAP analysis to ensure clinical transparency by identifying key predictive features like gamma band power and sample entropy.
3. Robust Performance: Achieved an average accuracy of 89.74% and an F1-score of 74.39% when evaluated on realistically imbalanced test sets.
4. Feature Engineering: Extracts 322 handcrafted features (time, frequency, and non-linear) from EEG spectrograms, reduced using Principal Component Analysis (PCA).
5. Imbalance Handling: Utilizes SMOTE (Synthetic Minority Oversampling Technique) exclusively on training folds to address the rarity of seizure events.

**Dataset**
This project uses the publicly available CHB-MIT Scalp EEG Database from PhysioNet.
1. Source: PhysioNet - CHB-MIT Scalp EEG Database
2. Patient Cohort: Long-term EEG recordings from 23 pediatric patients with intractable epilepsy.
3. Subset Used: Data from the first 10 patients (chb01-chb10) were selected for analysis.

Note: You must download the dataset separately from PhysioNet and place it in the designated data/raw folder for the scripts to run correctly.

**Technical Details**
1. Dependencies
The project requires the following libraries. You can install them using pip:
pip install -r requirements.txt

2. Model Architecture and Parameters
  a. Classifier: Light Gradient Boosting Machine (LGBM).
  b. Feature Set: 170 PCA-reduced features from an initial set of 322 handcrafted features.
  c. Evaluation: Stratified 10-fold cross-validation with SMOTE applied within the training folds.

Citation
If you use this code or methodology in your research, please cite the original paper: _To be added_

License
This project is licensed under the [Insert your preferred license, e.g., MIT License] - see ENSE.md file for details.
