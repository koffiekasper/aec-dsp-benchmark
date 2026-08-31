import math
import torch
from torch import stft as STFT
from torch import hamming_window 
import numpy as np

class OnlineFDNormalizer:
    def __init__(
        self,
        num_features,
        frame_shift=0.010,
        tau=3.0,
        eps=1e-8
    ):
        self.c = math.exp(-frame_shift / tau)
        self.eps = eps

        self.mean = torch.zeros(num_features)
        self.second_moment = torch.zeros(num_features)

    def process(self, x):
        self.mean = (
            self.c * self.mean
            + (1 - self.c) * x
        )

        self.second_moment = (
            self.c * self.second_moment
            + (1 - self.c) * x.square()
        )

        variance = (
            self.second_moment
            - self.mean.square()
        )

        variance = torch.clamp(
            variance,
            min=self.eps
        )

        return (
            x - self.mean
        ) / torch.sqrt(variance)

import torch


def STFTLogScaler(
    far_mic,
    far_lpb,
    n_fft,
    hop,
    window_l,
):
    far_mic = torch.as_tensor(far_mic, dtype=torch.float32)
    far_lpb = torch.as_tensor(far_lpb, dtype=torch.float32)

    batch_size = far_mic.shape[0]

    x = torch.cat(
        [far_mic, far_lpb],
        dim=0
    )

    window = torch.hamming_window(
        window_l,
        dtype=x.dtype,
        device=x.device
    )

    stft = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop,
        win_length=window_l,
        window=window,
        return_complex=True,
        center=False
    )

    power = stft.abs().square()

    log_power_db = 10 * torch.log10(
        power + 1e-12
    )

    log_power_db = torch.clamp(
        log_power_db,
        min=-120.0
    )

    mic = log_power_db[:batch_size]
    lpb = log_power_db[batch_size:]

    features = torch.cat(
        [mic, lpb],
        dim=1
    )

    return features
import torch

def STFTLogScaler(
    far_mic,
    far_lpb,
    near_mic,
    n_fft,
    hop,
    window_l,
):
    far_mic = torch.as_tensor(far_mic, dtype=torch.float32)
    far_lpb = torch.as_tensor(far_lpb, dtype=torch.float32)
    near_mic = torch.as_tensor(near_mic, dtype=torch.float32)

    B = far_mic.shape[0]

    x = torch.cat(
        [far_mic, far_lpb, near_mic],
        dim=0
    )

    window = torch.sqrt(
        torch.hann_window(
            window_l,
            periodic=True,
            dtype=x.dtype,
            device=x.device
        )
    )

    spec = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop,
        win_length=window_l,
        window=window,
        return_complex=True,
        center=False
    )

    mag = spec.abs()

    scaled = torch.log10(
        torch.clamp(mag.square(), min=1e-12)
    ) / 20.0

    mic_scaled = scaled[:B]
    lpb_scaled = scaled[B:2*B]

    features = torch.cat(
        [mic_scaled, lpb_scaled],
        dim=1
    )

    mic_mag = mag[:B]
    near_mag = mag[2*B:]

    return features, mic_mag, near_mag

def InverseSTFT(output,
                n_fft,
                hop,
                win_l):
    time_domain = torch.istft(
        input=output,
        n_fft = n_fft,
        hop_length = hop,
        win_length = win_l
    )
    return time_domain