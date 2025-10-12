import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)
from sklearn.preprocessing import label_binarize
from config import FEATURES_DIR, MODEL_SAVE_PATH, RESULTS_PATH, FOLDS

label_names = ['Interictal', 'Preictal', 'Ictal', 'Postictal']
n_classes = len(label_names)

# === Load Data ===
X = np.load(os.path.join(FEATURES_DIR, "X_features.npy"))
y = np.load(os.path.join(FEATURES_DIR, "y_labels.npy"))

all_preds, all_probs, all_true, fold_metrics = [], [], [], []

for fold in range(1, FOLDS+1):
    model_path = os.path.join(MODEL_SAVE_PATH, f"fold_{fold}.joblib")
    test_idx_path = os.path.join(MODEL_SAVE_PATH, f"fold_{fold}_test_idx.npy")
    if not os.path.exists(model_path) or not os.path.exists(test_idx_path):
        continue

    model = joblib.load(model_path)
    test_idx = np.load(test_idx_path)
    X_test, y_test = X[test_idx], y[test_idx]

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    all_preds.append(y_pred)
    all_probs.append(y_prob)
    all_true.append(y_test)

    fold_metrics.append({
        "fold": fold,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="macro"),
        "recall": recall_score(y_test, y_pred, average="macro"),
        "f1": f1_score(y_test, y_pred, average="macro"),
        "auc": roc_auc_score(label_binarize(y_test, classes=range(n_classes)), y_prob, multi_class='ovr')
    })

# === Combine Results ===
df_metrics = pd.DataFrame(fold_metrics)
y_true_all = np.concatenate(all_true)
y_pred_all = np.concatenate(all_preds)

# === Classification Report ===
clf_report = classification_report(y_true_all, y_pred_all, target_names=label_names, output_dict=True)
df_clf_report = pd.DataFrame(clf_report).T
df_clf_report.to_csv(os.path.join(RESULTS_PATH, "classification_report.csv"))

# === Confusion Matrix ===
cm = confusion_matrix(y_true_all, y_pred_all)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, "confusion_matrix.png"))
plt.close()

# === Fold-wise Metrics Bar Plot ===
metrics_melted = df_metrics.melt(id_vars="fold", var_name="Metric", value_name="Score")
plt.figure(figsize=(10, 6))
sns.barplot(data=metrics_melted, x="fold", y="Score", hue="Metric", palette="Set2")
plt.title("Performance Metrics per Fold")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, "fold_metrics_barplot.png"))
plt.close()

# === Boxplot of Metric Distributions ===
plt.figure(figsize=(10, 6))
sns.boxplot(data=metrics_melted, x="Metric", y="Score", palette="Set3")
plt.title("Metric Distribution Across Folds")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, "metric_boxplot.png"))
plt.close()

# === Save Summary ===
df_metrics.to_csv(os.path.join(RESULTS_PATH, "fold_metrics.csv"), index=False)
df_metrics.describe().to_csv(os.path.join(RESULTS_PATH, "fold_metrics_summary.csv"))
