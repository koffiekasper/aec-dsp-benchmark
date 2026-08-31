import numpy as np
import pandas as pd
import torch

from tqdm.auto import tqdm

from src.ERLE import erle


class AdaptiveFilterEvaluator:

    ERLE_TYPES = {
        "farend",
        "farend_move",
    }

    TYPE_ORDER = [
        "farend",
        "farend_move",
        "nearend",
        "synthetic_dt",
        "synthetic_dt_move",
    ]

    def __init__(
        self,
        model,
        dataset,
        n_fft=320,
        hop=160,
        window_l=320,
    ):
        self.model = model
        self.dataset = dataset

        self.n_fft = n_fft
        self.hop = hop
        self.window_l = window_l

        self.window = torch.sqrt(
            torch.hann_window(
                window_l,
                periodic=True,
                dtype=torch.float64,
            )
        )

    def _to_numpy(self, x):
        if torch.is_tensor(x):
            return (
                x.detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )

        return np.asarray(
            x,
            dtype=np.float64
        )

    def _spectral_mse(
        self,
        enhanced,
        target,
    ):
        enhanced = torch.as_tensor(
            enhanced,
            dtype=torch.float64
        )

        target = torch.as_tensor(
            target,
            dtype=torch.float64
        )

        length = min(
            enhanced.numel(),
            target.numel()
        )

        enhanced = enhanced[:length]
        target = target[:length]

        if length < self.window_l:
            return np.nan

        enhanced_spec = torch.stft(
            enhanced,
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.window_l,
            window=self.window,
            center=False,
            return_complex=True,
        )

        target_spec = torch.stft(
            target,
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.window_l,
            window=self.window,
            center=False,
            return_complex=True,
        )

        enhanced_mag = enhanced_spec.abs()
        target_mag = target_spec.abs()

        mse = torch.nn.functional.mse_loss(
            enhanced_mag,
            target_mag
        )

        return mse.item()

    def eval(self):

        rows = []

        model_name = self.model.__class__.__name__

        for sample in tqdm(
            self.dataset,
            desc=f"Evaluating {model_name}"
        ):
            scenario = sample["scenario"]

            mic = self._to_numpy(
                sample["mic"]
            )

            lpb = self._to_numpy(
                sample["lpb"]
            )

            target = self._to_numpy(
                sample["target"]
            )

            supervised = bool(
                sample.get(
                    "supervised",
                    True
                )
            )

            enhanced = self.model.fit_transform(
                lpb,
                mic
            )

            enhanced = self._to_numpy(
                enhanced
            )

            length = min(
                len(mic),
                len(target),
                len(enhanced),
            )

            mic = mic[:length]
            target = target[:length]
            enhanced = enhanced[:length]

            if scenario in self.ERLE_TYPES:
                erle_score = erle(
                    mic,
                    enhanced
                )
            else:
                erle_score = np.nan

            if supervised:
                mse_score = self._spectral_mse(
                    enhanced,
                    target
                )
            else:
                mse_score = np.nan

            rows.append({
                "Type": scenario,
                "ERLE": erle_score,
                "MSE": mse_score,
            })

        sample_df = pd.DataFrame(rows)

        result = (
            sample_df
            .groupby(
                "Type",
                as_index=False,
                sort=False
            )
            .agg({
                "ERLE": "mean",
                "MSE": "mean",
            })
        )

        order = {
            name: i
            for i, name
            in enumerate(self.TYPE_ORDER)
        }

        result["_order"] = (
            result["Type"]
            .map(order)
            .fillna(len(order))
        )

        result = (
            result
            .sort_values("_order")
            .drop(columns="_order")
            .reset_index(drop=True)
        )

        return result