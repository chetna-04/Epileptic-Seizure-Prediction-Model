import os
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
from config import FEATURES_DIR, MODEL_SAVE_PATH


# === Config ===
fold_to_plot = 1
X = np.load(os.path.join(FEATURES_DIR, "X_features.npy"))
shap_path = os.path.join(MODEL_SAVE_PATH, f"fold_{fold_to_plot}_shap.npy")
model_path = os.path.join(MODEL_SAVE_PATH, f"fold_{fold_to_plot}.joblib")
test_idx_path = os.path.join(MODEL_SAVE_PATH, f"fold_{fold_to_plot}_test_idx.npy")

# === Load model and SHAP values ===
print(f"[📦] Loading model from: {model_path}")
model = joblib.load(model_path)
shap_values = np.load(shap_path, allow_pickle=True)
test_idx = np.load(test_idx_path)
X_test = X[test_idx]

# === Plot SHAP summary ===
print("[📊] Plotting SHAP summary...")

if isinstance(shap_values, list):  # multiclass
    for class_idx, class_shap in enumerate(shap_values):
        plt.figure()
        shap.summary_plot(class_shap, X_test, show=False)
        plt.title(f"SHAP Summary - Class {class_idx}")
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_SAVE_PATH, f"shap_summary_class_{class_idx}.png"))
        plt.close()
else:
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Summary")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_SAVE_PATH, "shap_summary.png"))
    plt.close()

print("[✅] SHAP visualization complete.")
