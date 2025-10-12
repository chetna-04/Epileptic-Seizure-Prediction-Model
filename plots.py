import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from scipy.signal import welch
from mne.io import read_raw_edf
import joblib
from config import FEATURES_DIR, DATA_DIR, PREPROCESSED_DIR, MODEL_SAVE_PATH, RESULTS_PATH, SAMPLING_RATE

# ========== CONFIG ==========
PLOT_SELECTION = [1, 5]  # Choose from 1 to 6

label_map = {0: 'Interictal', 1: 'Preictal', 2: 'Ictal', 3: 'Postictal'}
label_colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c']  # Blue, Orange, Red, Green
channels = {1:'FP1-F7', 2:'F7-T7', 3:'T7-P7', 4:'P7-01', 5:'FP1-F3', 6:'F3-C3', 7:'C3-P3', 8:'P3-01', 9:'FP2-F', 10:'F4-C4', 
            11:'C4-P4', 12:'P4-02', 13:'FP2-F8', 14:'F8-T8', 15:'T8-P8', 16:'P8-02', 17:'FZ-CZ', 18:'CZ-PZ', 19:'P7-T7', 20:'T7-FT9', 
            21:'FT9-FT10', 22:'FT10-T8', 23:'T8-P8'}

# ========== 1. SNR Before vs After ==========
if 1 in PLOT_SELECTION:
    print("\n------SNR Before vs After Preprocessing------")

    def compute_snr(signal, noise):
        return 10 * np.log10(np.mean(signal ** 2) / (np.mean(noise ** 2) + 1e-9))

    snr_raw_all = []
    snr_pre_all = []
    patient_ids = []

    for patient_id in sorted(os.listdir(DATA_DIR)):
        edf_dir = os.path.join(DATA_DIR, patient_id)
        spec_dir = os.path.join(PREPROCESSED_DIR, patient_id, "spectrograms")

        edf_files = sorted([f for f in os.listdir(edf_dir) if f.endswith('.edf')])
        npy_files = sorted([f for f in os.listdir(spec_dir) if f.endswith('.npy')])

        if not edf_files or not npy_files:
            continue

        edf_path = os.path.join(edf_dir, edf_files[0])
        npy_path = os.path.join(spec_dir, npy_files[0])

        try:
            raw = read_raw_edf(edf_path, preload=True, verbose=False)
            raw.pick_channels(raw.ch_names[:23])
            raw_data = raw.get_data()
            raw_segment = raw_data[:, :SAMPLING_RATE * 10]
            noise_raw = raw_segment - np.mean(raw_segment, axis=1, keepdims=True)
            raw_snr = compute_snr(raw_segment, noise_raw)

            preprocessed = np.load(npy_path)
            pre_segment = preprocessed[0][:23]
            noise_pre = pre_segment - np.mean(pre_segment, axis=1, keepdims=True)
            pre_snr = compute_snr(pre_segment, noise_pre)

            snr_raw_all.append(raw_snr)
            snr_pre_all.append(pre_snr)
            patient_ids.append(patient_id)

        except Exception as e:
            print(f"Error processing {patient_id}: {e}")

    plt.figure(figsize=(10, 6))
    plt.plot(patient_ids, snr_raw_all, marker='o', linestyle='-', label='Raw EEG', color='gray')
    plt.plot(patient_ids, snr_pre_all, marker='s', linestyle='-', label='Preprocessed EEG', color='teal')
    plt.xticks(rotation=45)
    plt.ylabel('SNR (dB)', fontsize=12)
    plt.xlabel('Patient ID', fontsize=12)
    plt.title('SNR Distribution Across Patients (Raw vs Preprocessed)', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'snr_distribution_lineplot.png'), dpi=300)
    plt.close()

# ========== 2. Feature Distribution by Class ==========
if 2 in PLOT_SELECTION:
    print("\n------Feature Distribution by Class------")
    X = np.load(os.path.join(FEATURES_DIR, 'X_features.npy'))
    y = np.load(os.path.join(FEATURES_DIR, 'y_labels.npy'))

    df = pd.DataFrame(X[:, :5], columns=[f"F{i}" for i in range(5)])
    df['Class'] = [label_map[val] for val in y]

    sns.pairplot(df, hue='Class', palette=label_colors[:4])
    plt.suptitle('Feature Distributions by Class', fontsize=14)
    plt.savefig(os.path.join(RESULTS_PATH, 'feature_distributions.png'))
    plt.close()

# ========== 3. PCA ==========
if 3 in PLOT_SELECTION:
    print("\n------PCA Plot------")
    df = pd.DataFrame(X[:, 13:15], columns=['PC1', 'PC2'])
    df['Class'] = [label_map[val] for val in y]
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='PC1', y='PC2', hue='Class', palette=label_colors)
    plt.title("Features after PCA")
    plt.savefig(os.path.join(RESULTS_PATH, 'pca_plot.png'))
    plt.close()

# ========== 4. Class Histogram Before and After SMOTE ==========
if 4 in PLOT_SELECTION:
    print("\n------Class Distribution Before/After SMOTE------")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    plt.figure(figsize=(8, 5))
    before = pd.Series(y).map(label_map).value_counts()
    after = pd.Series(y_res).map(label_map).value_counts()

    width = 0.35
    idx = np.arange(len(before))
    plt.bar(idx - width/2, before.values, width, label='Before SMOTE', color='gray')
    plt.bar(idx + width/2, after.values, width, label='After SMOTE', color='teal')
    plt.xticks(idx, before.index)
    plt.ylabel("Samples")
    plt.legend()
    plt.title("Class Distribution Before and After SMOTE")
    plt.savefig(os.path.join(RESULTS_PATH, 'smote_class_distribution.png'))
    plt.close()

# ========== 5. PSD ==========
if 5 in PLOT_SELECTION:
    print("\n------PSD Comparison------")
    plt.figure(figsize=(10, 6))
    for ch in range(2):
        f_raw, psd_raw = welch(raw_data[ch, :], fs=SAMPLING_RATE, nperseg=256)
        f_pre, psd_pre = welch(preprocessed[0][ch].flatten(), fs=SAMPLING_RATE)
        ch_label = channels[ch+1]
        plt.semilogy(f_raw, psd_raw, label=f'Raw channel = {ch_label}')
        plt.semilogy(f_pre, psd_pre, linestyle='--', label=f'Preprocessed channel = {ch_label}')

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.title('PSD Before vs After Preprocessing')
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'psd_comparison.png'), bbox_inches='tight')
    plt.close()

# ========== 6. ROC Curves ==========
if 6 in PLOT_SELECTION:
    print("\n------ROC Curves (Mean ± Std)------")
    n_classes = 4
    y_binarized = label_binarize(y, classes=[0, 1, 2, 3])
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    mean_fpr = np.linspace(0, 1, 100)
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        tprs = []
        aucs = []

        for fold in range(1, 11):
            model_path = os.path.join(MODEL_SAVE_PATH, f"fold_{fold}.joblib")
            idx_path = os.path.join(MODEL_SAVE_PATH, f"fold_{fold}_test_idx.npy")
            if not (os.path.exists(model_path) and os.path.exists(idx_path)):
                continue
            model = joblib.load(model_path)
            test_idx = np.load(idx_path)
            X_test_fold = X[test_idx]
            y_test_fold = y[test_idx]
            y_score = model.predict_proba(X_test_fold)

            fpr, tpr, _ = roc_curve(y_test_fold == i, y_score[:, i])
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)

        mean_tpr = np.mean(tprs, axis=0)
        std_tpr = np.std(tprs, axis=0)
        mean_auc = auc(mean_fpr, mean_tpr)
        label = f"{label_map[i]} (AUC = {mean_auc:.2f})"
        label_name = label_map[i]
        label_index = list(label_map.values()).index(label_name)
        color = label_colors[label_index]
        plt.plot(mean_fpr, mean_tpr, label=label, color=color)
        plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, alpha=0.2, color=color)

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (Mean ± Std across folds)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'roc_curves_mean_std.png'))
    plt.close()

print("\n✅ All selected plots generated and saved to:", RESULTS_PATH)