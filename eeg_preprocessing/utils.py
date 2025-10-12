import mne
import numpy as np
import pandas as pd

def load_edf_file(path):
    raw = mne.io.read_raw_edf(path, preload=True)
    raw.pick_types(eeg=True)
    return raw

def get_annotations_from_csv(csv_path, patient_id, file_name):
    df = pd.read_csv(csv_path)
    df = df[(df['patient_id'] == patient_id) & (df['file_name'] == file_name)]
    return [(float(row['seizure_onset']), float(row['seizure_duration']), row['label']) for _, row in df.iterrows()]

def label_seizure_stages(annotations, sfreq, total_duration_sec):
    """
    Assigns per-sample labels based on seizure annotations.
    Labels:
        0 - interictal
        1 - preictal (1h before seizure)
        2 - ictal (during seizure)
        3 - postictal (1h after seizure)
    """
    labels = np.zeros(int(sfreq * total_duration_sec), dtype=int)
    seizures = sorted(annotations, key=lambda x: x[0])  # sort by onset

    last_seizure_end = -np.inf
    for i, (onset_sec, duration_sec, _) in enumerate(seizures):
        end_sec = onset_sec + duration_sec

        # Skip non-leading seizures
        if onset_sec - last_seizure_end < 3600:
            continue  # skip if <1hr since last seizure

        # Ictal
        start_idx = int(onset_sec * sfreq)
        end_idx = int(end_sec * sfreq)
        labels[start_idx:end_idx] = 2

        # Preictal (1 hour before onset)
        pre_start = int(max(0, (onset_sec - 3600) * sfreq))
        pre_end = start_idx
        labels[pre_start:pre_end] = 1

        # Postictal (1 hour after end)
        post_start = end_idx
        post_end = int(min(labels.size, (end_sec + 3600) * sfreq))
        labels[post_start:post_end] = 3

        last_seizure_end = end_sec

    return labels
