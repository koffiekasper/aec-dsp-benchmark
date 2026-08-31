from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from pathlib import Path
import random
import numpy as np
import pandas as pd
from scipy.io import wavfile


class AECDataset(Dataset):
    FILES = {
        "far_lpb": "_farend_singletalk_lpb.wav",
        "far_mic": "_farend_singletalk_mic.wav",
        "far_move_lpb": "_farend_singletalk_with_movement_lpb.wav",
        "far_move_mic": "_farend_singletalk_with_movement_mic.wav",
        "near_mic": "_nearend_singletalk_mic.wav",
        "dt_lpb": "_doubletalk_lpb.wav",
        "dt_mic": "_doubletalk_mic.wav",
        "dt_move_lpb": "_doubletalk_with_movement_lpb.wav",
        "dt_move_mic": "_doubletalk_with_movement_mic.wav",
    }

    def __init__(
        self,
        path,
        seed=None,
        sample_rate=16000,
        chunk_seconds=4,
        ser_range=(-10, 10),
    ):
        self.path = Path(path)
        self.crop_n = sample_rate * chunk_seconds
        self.ser_range = ser_range

        if seed is not None:
            random.seed(seed)

        self.df = self._build_df()

    def __len__(self):
        return len(self.df)

    def _build_df(self):
        grouped = {}

        for p in self.path.glob("*.wav"):
            for key, ending in self.FILES.items():
                if p.name.lower().endswith(ending):
                    guid = p.name[:-len(ending)]
                    grouped.setdefault(guid, {})[key] = p
                    break

        scenarios = [
            # name, mic, lpb, near, echo, supervised
            ("farend", "far_mic", "far_lpb", None, None, True),
            ("farend_move", "far_move_mic", "far_move_lpb", None, None, True),
            ("nearend", "near_mic", None, "near_mic", None, True),
            ("synthetic_dt", None, "far_lpb", "near_mic", "far_mic", True),
            ("synthetic_dt_move", None, "far_move_lpb", "near_mic", "far_move_mic", True),
            ("real_dt", "dt_mic", "dt_lpb", None, None, False),
            ("real_dt_move", "dt_move_mic", "dt_move_lpb", None, None, False),
        ]

        rows = []

        for guid, g in grouped.items():
            for scenario, mic, lpb, near, echo, supervised in scenarios:
                needed = [x for x in [mic, lpb, near, echo] if x]

                if all(x in g for x in needed):
                    rows.append({
                        "guid": guid,
                        "scenario": scenario,
                        "mic": g.get(mic),
                        "lpb": g.get(lpb),
                        "near": g.get(near),
                        "echo": g.get(echo),
                        "supervised": supervised,
                    })

        return pd.DataFrame(rows)

    def _load(self, path):
        _, x = wavfile.read(path)

        dtype = x.dtype
        x = x.astype(np.float32)

        if np.issubdtype(dtype, np.integer):
            x /= 32768.0

        if x.ndim > 1:
            x = x.mean(axis=1)

        return torch.from_numpy(x)

    def _crop(self, *xs):
        n = min(map(len, xs))
        xs = [x[:n] for x in xs]

        if n >= self.crop_n:
            start = random.randint(0, n - self.crop_n)
            return [x[start:start + self.crop_n] for x in xs]

        return [
            F.pad(x, (0, self.crop_n - n))
            for x in xs
        ]

    def _mix(self, near, echo):
        ser_db = random.uniform(*self.ser_range)

        near_rms = near.square().mean().sqrt()
        echo_rms = echo.square().mean().sqrt()

        alpha = near_rms / (
            echo_rms * 10 ** (ser_db / 20) + 1e-8
        )

        return near + alpha * echo

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        s = row["scenario"]

        if s == "nearend":
            near, = self._crop(self._load(row["near"]))
            mic = target = near
            lpb = torch.zeros_like(near)

        elif s.startswith("farend"):
            mic, lpb = self._crop(
                self._load(row["mic"]),
                self._load(row["lpb"]),
            )
            target = torch.zeros_like(mic)

        elif s.startswith("synthetic"):
            near, echo, lpb = self._crop(
                self._load(row["near"]),
                self._load(row["echo"]),
                self._load(row["lpb"]),
            )
            mic = self._mix(near, echo)
            target = near

        else:
            mic, lpb = self._crop(
                self._load(row["mic"]),
                self._load(row["lpb"]),
            )
            target = torch.zeros_like(mic)

        return {
            "mic": mic,
            "lpb": lpb,
            "target": target,
            "scenario": s,
            "supervised": row["supervised"],
            "guid": row["guid"],
        }