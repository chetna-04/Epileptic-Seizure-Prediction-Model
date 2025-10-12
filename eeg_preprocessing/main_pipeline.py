import os
import numpy as np
from config import DATA_DIR, PREPROCESSED_DIR, ANNOTATION_CSV_PATH, WINDOW_SIZE, STEP_SIZE, NUM_CHANNELS
from filtering import apply_bandpass_filter, apply_notch_filter, remove_dc_offset
from artifact_removal import remove_all_artifacts
from segmentation import sliding_window
from signal_transform import compute_stft
from normalize import apply_minmax_scaling, standardize_channels
from utils import load_edf_file, label_seizure_stages, get_annotations_from_csv

def process_and_save_file(edf_path, save_path, csv_path, patient_id, file_name, win_size=WINDOW_SIZE, step_size=STEP_SIZE):
    try:
        print(f"\n[INFO] Processing: {edf_path}")
        raw = load_edf_file(edf_path)
        raw = standardize_channels(raw, NUM_CHANNELS)
        fs = int(raw.info['sfreq'])

        # Apply filters + artifact removal here...
        raw = apply_bandpass_filter(raw, 0.5, 30.0)
        raw = apply_notch_filter(raw, [60, 120], 6)
        raw = remove_dc_offset(raw)
        raw = remove_all_artifacts(raw, use_ica=True, remove_emg=True)

        data = raw.get_data()
        n_samples = data.shape[1]
        total_duration = n_samples / fs

        annotations = get_annotations_from_csv(csv_path, patient_id, file_name)
        labels = label_seizure_stages(annotations, fs, total_duration)

        segments, segment_labels = sliding_window(data, labels, win_size, step_size, fs)
        spectrograms = np.array([compute_stft(seg, fs) for seg in segments])
        spectrograms = apply_minmax_scaling(spectrograms)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path.replace(".edf", "_X.npy"), spectrograms)
        np.save(save_path.replace(".edf", "_y.npy"), segment_labels)
        print(f"[✔] Saved: {save_path.replace('.edf', '_X.npy')}")

    except Exception as e:
        print(f"[✖] Failed: {edf_path}, Error: {e}")

def process_all_patients():
    csv_path = ANNOTATION_CSV_PATH

    for patient_dir in sorted(os.listdir(DATA_DIR)):
        patient_path = os.path.join(DATA_DIR, patient_dir)
        if not os.path.isdir(patient_path):
            continue

        for file in os.listdir(patient_path):
            if file.endswith(".edf"):
                edf_path = os.path.join(patient_path, file)
                save_dir = os.path.join(PREPROCESSED_DIR, patient_dir)
                save_path = os.path.join(save_dir, file)

                process_and_save_file(edf_path, save_path, csv_path, patient_dir, file)

if __name__ == "__main__":
    process_all_patients()
