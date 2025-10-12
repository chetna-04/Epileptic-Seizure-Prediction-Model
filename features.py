import os
import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.signal import welch
from antropy import sample_entropy, higuchi_fd, hjorth_params
import matplotlib.pyplot as plt
import multiprocessing as mp
from config import PREPROCESSED_DIR, FEATURES_DIR, SAMPLING_RATE, NUM_CHANNELS

FREQ_BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (12, 30),
    "gamma": (30, 45),
}

# === Feature Functions ===

def bandpower(psd, freqs, band):
    idx = np.logical_and(freqs >= band[0], freqs <= band[1])
    return np.sum(psd[idx]) if np.any(idx) else 0.0

def extract_features_from_spectrogram(spectrogram, fs):
    n_channels = NUM_CHANNELS
    features = []

    for ch in range(n_channels):
        ch_spec = spectrogram[ch]
        ch_features = []

        avg_power = np.mean(ch_spec, axis=1)
        total_power = np.sum(avg_power)

        freqs = np.linspace(0, fs // 2, ch_spec.shape[0])
        for band in FREQ_BANDS.values():
            abs_power = bandpower(avg_power, freqs, band)
            rel_power = abs_power / total_power if total_power > 0 else 0.0
            ch_features.extend([abs_power, rel_power])

        flat_spec = ch_spec.flatten()
        ch_features.append(sample_entropy(flat_spec))
        ch_features.append(higuchi_fd(flat_spec))

        time_avg = np.mean(ch_spec, axis=0)
        mobility, complexity = hjorth_params(time_avg)
        ch_features.extend([mobility, complexity])

        features.extend(ch_features)

    return features  # (23 * 14)

# === Multiprocessing File Handler ===

def process_file_pair(paths):
    x_path, y_path = paths
    features = []
    labels = []
    bandpowers = []

    try:
        X = np.load(x_path)
        y = np.load(y_path)

        for i in range(X.shape[0]):
            spec = X[i]
            label = y[i]

            f = extract_features_from_spectrogram(spec, fs=SAMPLING_RATE)

            ch_spec = spec[0]
            freqs = np.linspace(0, SAMPLING_RATE // 2, ch_spec.shape[0])
            avg_power = np.mean(ch_spec, axis=1)
            bp = [bandpower(avg_power, freqs, band) for band in FREQ_BANDS.values()]

            features.append(f)
            labels.append(label)
            bandpowers.append((bp, label))

    except Exception as e:
        print(f"[⚠️] Failed to process file {x_path}: {e}")

    return features, labels, bandpowers

# === Main Pipeline ===

def main():
    all_features = []
    all_labels = []
    all_bandpowers = []

    # === Gather all (X, y) file pairs ===
    file_pairs = []
    for patient_dir in sorted(os.listdir(PREPROCESSED_DIR)):
        p_dir = os.path.join(PREPROCESSED_DIR, patient_dir)
        if not os.path.isdir(p_dir): continue

        spec_dir = os.path.join(p_dir, "spectrograms")
        label_dir = os.path.join(p_dir, "labels")

        for x_path in sorted(glob(os.path.join(spec_dir, "chb*_X.npy"))):
            y_path = os.path.join(label_dir, os.path.basename(x_path).replace("X", "y"))
            if os.path.exists(y_path):
                file_pairs.append((x_path, y_path))

    print(f"[🚀] Processing {len(file_pairs)} files using 5 cores...")

    # === Parallel file loading and feature extraction ===
    with mp.Pool(processes=5) as pool:
        results = list(tqdm(pool.imap(process_file_pair, file_pairs), total=len(file_pairs)))

    # === Combine results from all processes ===
    for f_list, l_list, bp_list in results:
        all_features.extend(f_list)
        all_labels.extend(l_list)
        all_bandpowers.extend(bp_list)

    all_features = np.array(all_features)
    all_labels = np.array(all_labels)
    print(f"[✔] Raw Feature shape: {all_features.shape}")

    # === Visualization: Bandpower Distribution ===

    bandpowers_np = np.array([bp for bp, _ in all_bandpowers])
    labels_np = np.array([lbl for _, lbl in all_bandpowers])
    band_names = list(FREQ_BANDS.keys())

    plt.figure(figsize=(12, 6))
    for i, band in enumerate(band_names):
        plt.subplot(1, len(band_names), i + 1)
        plt.boxplot([bandpowers_np[labels_np == 0, i], bandpowers_np[labels_np == 1, i]], labels=["Interictal", "Preictal"])
        plt.title(band)
        plt.xticks(rotation=45)
        plt.tight_layout()
    plt.suptitle("Bandpower Distributions")
    plt.tight_layout()
    plt.savefig(os.path.join(FEATURES_DIR, "bandpower_distribution.png"))
    plt.close()

    # === PCA for Dimensionality Reduction ===

    print("[🔍] Applying PCA to reduce dimensionality...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(all_features)

    pca = PCA(n_components=0.99)  # retain 99% variance
    X_pca = pca.fit_transform(X_scaled)
    print(f"[✅] Reduced Feature shape after PCA: {X_pca.shape}")

    # === Save Features ===

    np.save(os.path.join(FEATURES_DIR, "X_features.npy"), X_pca)
    np.save(os.path.join(FEATURES_DIR, "y_labels.npy"), all_labels)
    print(f"[💾] Saved: X_features.npy {X_pca.shape}, y_labels.npy {all_labels.shape}")

if __name__ == "__main__":
    main()
