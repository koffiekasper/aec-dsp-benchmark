from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from pathlib import Path
import random
import pandas as pd
from scipy.io import wavfile


class FarEndSingleTalkDataset(Dataset):
    def __init__(
        self,
        path,
        seed=None,
        n=-1,
        sample_rate=16000,
        chunk_seconds=4
    ):
        self.path = Path(path)

        if seed is not None:
            random.seed(seed)

        self.n = n
        self.sample_rate = sample_rate
        self.crop_n = sample_rate * chunk_seconds

        self.df = pd.DataFrame(
            columns=[
                "farend_mic_path",
                "farend_lpb_path",
                "nearend_mic_path"
            ]
        )

        self._populate_df()

    def __len__(self):
        return len(self.df)

    def _populate_df(self):
        farend_mic_paths = sorted(
            self.path.glob("*farend_singletalk_mic.wav")
        )

        pairs = []

        for farend_mic_path in farend_mic_paths:
            farend_lpb_path = farend_mic_path.with_name(
                farend_mic_path.name.replace(
                    "farend_singletalk_mic.wav",
                    "farend_singletalk_lpb.wav"
                )
            )

            nearend_mic_path = farend_mic_path.with_name(
                farend_mic_path.name.replace(
                    "farend_singletalk_mic.wav",
                    "nearend_singletalk_mic.wav"
                )
            )

            if farend_lpb_path.exists() and nearend_mic_path.exists():
                pairs.append({
                    "farend_mic_path": farend_mic_path,
                    "farend_lpb_path": farend_lpb_path,
                    "nearend_mic_path": nearend_mic_path
                })

        if self.n != -1:
            pairs = random.sample(
                pairs,
                min(self.n, len(pairs))
            )

        self.df = pd.DataFrame(pairs)

    def _load_audio(self, idx):
        row = self.df.iloc[idx]

        _, farend_mic = wavfile.read(row["farend_mic_path"])
        _, farend_lpb = wavfile.read(row["farend_lpb_path"])
        _, nearend_mic = wavfile.read(row["nearend_mic_path"])

        farend_mic = torch.from_numpy(
            farend_mic.astype("float32") / 32768.0
        )

        farend_lpb = torch.from_numpy(
            farend_lpb.astype("float32") / 32768.0
        )

        nearend_mic = torch.from_numpy(
            nearend_mic.astype("float32") / 32768.0
        )

        n = min(
            len(farend_mic),
            len(farend_lpb),
            len(nearend_mic)
        )

        return (
            farend_mic[:n],
            farend_lpb[:n],
            nearend_mic[:n]
        )

    def __getitem__(self, idx):
        farend_mic, farend_lpb, nearend_mic = self._load_audio(idx)

        n = len(farend_mic)

        if n >= self.crop_n:
            start = random.randint(0, n - self.crop_n)
            end = start + self.crop_n

            farend_mic = farend_mic[start:end]
            farend_lpb = farend_lpb[start:end]
            nearend_mic = nearend_mic[start:end]

        else:
            pad_n = self.crop_n - n

            farend_mic = F.pad(farend_mic, (0, pad_n))
            farend_lpb = F.pad(farend_lpb, (0, pad_n))
            nearend_mic = F.pad(nearend_mic, (0, pad_n))

        return {
            "farend_mic": farend_mic,
            "farend_lpb": farend_lpb,
            "nearend_mic": nearend_mic,
            "length": self.crop_n,
            "index": idx
        }