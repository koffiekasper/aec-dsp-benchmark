import numpy as np

def normalize_aec_triplet(
    far_lpb: np.ndarray,
    far_mic: np.ndarray,
    near_mic: np.ndarray,
    target_rms: float = 0.05,
    eps: float = 1e-12,
):
    far_lpb = np.asarray(far_lpb, dtype=np.float32)
    far_mic = np.asarray(far_mic, dtype=np.float32)
    near_mic = np.asarray(near_mic, dtype=np.float32)

    lpb_rms = np.sqrt(np.mean(far_lpb**2) + eps)

    gain = target_rms / lpb_rms

    return (
        far_lpb * gain,
        far_mic * gain,
        near_mic * gain,
        gain,
    )