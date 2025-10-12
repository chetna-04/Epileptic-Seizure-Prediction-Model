import os
import numpy as np
import joblib
import shap
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from imblearn.over_sampling import SMOTE
from config import FEATURES_DIR, MODEL_SAVE_PATH, FOLDS

# === Load Data ===
X = np.load(os.path.join(FEATURES_DIR, "X_features.npy"))
y = np.load(os.path.join(FEATURES_DIR, "y_labels.npy"))

print(f"[📊] Data shape: X={X.shape}, y={y.shape}")
print("Any NaNs in X?", np.isnan(X).any())
print("All features constant?", np.all(X == X[0], axis=0).sum(), "/", X.shape[1])
print("Number of non-zero features per sample (mean):", np.count_nonzero(X, axis=1).mean())

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
fold_results = []

# === Cross-Validation ===
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(tqdm(list(skf.split(X, y)), desc="[🚀] Cross-Validation Progress"), 1):
    model_path = os.path.join(MODEL_SAVE_PATH, f"fold_{fold}.joblib")
    idx_path = os.path.join(MODEL_SAVE_PATH, f"fold_{fold}_test_idx.npy")

    if os.path.exists(model_path) and os.path.exists(idx_path):
        print(f"[⏭️] Fold {fold} already completed (found {model_path} and {idx_path}). Skipping...")
        continue

    print(f"\n[📁] Fold {fold}")

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    np.save(idx_path, test_idx)  # Save test indices early

    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    model = LGBMClassifier(
        n_estimators=1700,
        learning_rate=0.05,
        class_weight='balanced',
        objective='multiclass',
        num_class=4,
        random_state=42,
        n_jobs=5
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='multi_logloss',
        callbacks=[
            early_stopping(stopping_rounds=20),
            log_evaluation(period=50)
        ]
    )

    joblib.dump(model, model_path)
    print(f"[💾] Saved model to: {model_path}")
    print(f"[🧠] Best iteration for fold {fold}: {model.best_iteration_}")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)  # shape: (n_samples, 4)

    # === Metrics ===
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    try:
        auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
    except ValueError:
        auc = 0.0

    fold_results.append({
        "fold": fold,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc
    })

    # === SHAP ===
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)  # list of arrays (one per class)
    shap_path = os.path.join(MODEL_SAVE_PATH, f"fold_{fold}_shap.npy")
    np.save(shap_path, shap_values)
    print(f"[📉] Saved SHAP values to: {shap_path}")

# === Final Summary ===
if fold_results:
    print("\n[📈] Completed Fold Metrics:")
    for res in fold_results:
        print(f"Fold {res['fold']}: Acc={res['accuracy']:.4f}, Prec={res['precision']:.4f}, "
              f"Rec={res['recall']:.4f}, F1={res['f1']:.4f}, AUC={res['auc']:.4f}")

    avg = {k: np.mean([r[k] for r in fold_results]) for k in fold_results[0] if k != "fold"}
    print("\n[🔍] Average Metrics Across Folds:")
    for k, v in avg.items():
        print(f"{k.capitalize()}: {v:.4f}")
else:
    print("[✅] All folds were already completed. Nothing to train.")
