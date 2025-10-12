from mne.preprocessing import ICA, annotate_muscle_zscore

def remove_emg_noise_automatic(raw, threshold=4.0):
    """
    Automatically detect and annotate muscle (EMG) artifacts using MNE's z-score method.
    """
    print("[INFO] Detecting muscle artifacts using z-score...")
    annot_muscle, scores = annotate_muscle_zscore(
        raw, 
        threshold=threshold,
        ch_type="eeg",
        filter_freq=(110, 127)
    )
    raw.set_annotations(raw.annotations + annot_muscle)
    return raw

def remove_eog_artifacts_ica(raw, n_components=15):
    """
    Remove EOG artifacts using ICA.
    If no EOG channels are present, skip ICA-based removal.
    """
    print("[INFO] Running ICA for EOG artifact removal...")

    ica = ICA(n_components=n_components, random_state=97, max_iter=800, method='fastica')
    ica.fit(raw)

    try:
        eog_inds, scores = ica.find_bads_eog(raw)
    except RuntimeError as e:
        if "No EOG channel" in str(e):
            print("[INFO] No EOG channels found. Skipping ICA exclusion.")
            return raw
        else:
            raise e

    if not eog_inds:
        print("[INFO] No EOG components detected. Skipping ICA exclusion.")
        return raw

    ica.exclude = eog_inds
    print(f"[INFO] ICA: Excluded {len(eog_inds)} EOG components.")

    return ica.apply(raw.copy())

def remove_all_artifacts(raw, use_ica=True, remove_emg=True):
    """
    Apply all artifact removal steps with configuration.
    """
    if remove_emg:
        raw = remove_emg_noise_automatic(raw)

    if use_ica:
        raw = remove_eog_artifacts_ica(raw)
    return raw
