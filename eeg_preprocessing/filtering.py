def apply_bandpass_filter(raw, l_freq=0.5, h_freq=30.0):
    """
    Apply a band-pass FIR filter to isolate relevant EEG frequency bands.
    Default range is 0.5–30 Hz, but can be changed to include gamma (e.g., up to 128 Hz).
    """
    raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin')
    return raw

def apply_notch_filter(raw, freqs=[60, 120], band_width=6):
    """
    Apply band-stop filtering centered at specified frequencies with a defined band width.
    For example, for 60Hz and width=6 → notch 57–63Hz.
    """
    for freq in freqs:
        raw = raw.filter(
            l_freq=freq - band_width / 2,
            h_freq=freq + band_width / 2,
            method='fir',
            phase='zero',
            fir_design='firwin',
            verbose=False
        )
    return raw

def remove_dc_offset(raw):
    """
    Remove 0 Hz (DC) component by subtracting the mean of each channel.
    """
    raw._data = raw._data - raw._data.mean(axis=1, keepdims=True)
    return raw
