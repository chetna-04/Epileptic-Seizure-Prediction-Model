from sklearn.preprocessing import MinMaxScaler

def apply_minmax_scaling(data, feature_range=(0, 1)):
    scaler = MinMaxScaler(feature_range=feature_range)
    flat_data = data.reshape(-1, data.shape[-1])
    scaled_data = scaler.fit_transform(flat_data)
    return scaled_data.reshape(data.shape)

def standardize_channels(raw, expected_channels):
    """
    Ensure consistent channel naming and count.
    - Rename duplicates
    - Drop extras
    """
    # Make channel names unique if needed
    if len(set(raw.ch_names)) != len(raw.ch_names):
        raw.rename_channels({
            name: f"{name}_{i}" for i, name in enumerate(raw.ch_names)
            if raw.ch_names.count(name) > 1
        })

    # Truncate if more than expected
    if len(raw.ch_names) > expected_channels:
        raw.pick_channels(raw.ch_names[:expected_channels])
    return raw

