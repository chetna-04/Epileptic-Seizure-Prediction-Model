import numpy as np
from scipy.signal import stft

def compute_stft(data, fs, n_fft=256, noverlap=128):
    f, t, Zxx = stft(data, fs=fs, nperseg=n_fft, noverlap=noverlap, axis=1)
    return np.abs(Zxx)
