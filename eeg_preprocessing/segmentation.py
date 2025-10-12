import numpy as np

def sliding_window(data, labels, window_size, step_size, fs):
    """
    Segment EEG into windows with corresponding labels.
    Multiclass labeling: 0=interictal, 1=preictal, 2=ictal, 3=postictal
    """
    n_samples = data.shape[1]
    win_len = int(window_size * fs)
    step_len = int(step_size * fs)

    windows = []
    window_labels = []

    for start in range(0, n_samples - win_len + 1, step_len):
        end = start + win_len
        segment = data[:, start:end]
        window_label = int(np.round(np.median(labels[start:end])))
        windows.append(segment)
        window_labels.append(window_label)

    return np.array(windows), np.array(window_labels)
