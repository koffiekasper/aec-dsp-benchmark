import random
import torch

from src.XiaModel.PreProcess import STFTLogScaler


def sample_model_outputs(
    test_dataset,
    model,
    n=5,
    n_fft=320,
    hop=160,
    window_l=320,
    device=None,
):
    if device is None:
        device = next(model.parameters()).device

    n = min(n, len(test_dataset))
    indices = random.sample(
        range(len(test_dataset)),
        n
    )

    model.eval()
    outputs = []

    window = torch.sqrt(
        torch.hann_window(
            window_l,
            periodic=True,
            device=device,
            dtype=torch.float32,
        )
    )

    hidden_size = model.gru1.hidden_size

    with torch.inference_mode():

        for idx in indices:
            sample = test_dataset[idx]

            if "mic" in sample:
                mic = sample["mic"].to(device)
                lpb = sample["lpb"].to(device)
                target = sample["target"].to(device)

                scenario = sample.get(
                    "scenario",
                    "unknown"
                )

                supervised = sample.get(
                    "supervised",
                    True
                )

                guid = sample.get(
                    "guid",
                    None
                )

            else:
                far_mic = sample["farend_mic"].to(device)
                lpb = sample["farend_lpb"].to(device)
                target = sample["nearend_mic"].to(device)

                mic = target + far_mic

                scenario = "synthetic_doubletalk"
                supervised = True
                guid = None

            features, mic_mag, target_mag = STFTLogScaler(
                far_mic=mic.unsqueeze(0),
                far_lpb=lpb.unsqueeze(0),
                near_mic=target.unsqueeze(0),
                n_fft=n_fft,
                hop=hop,
                window_l=window_l,
            )

            features = features.to(device)

            T = features.size(2)

            h01 = torch.zeros(
                1,
                1,
                hidden_size,
                device=device,
            )

            h02 = torch.zeros(
                1,
                1,
                hidden_size,
                device=device,
            )

            masks = []

            for t in range(T):
                x_t = (
                    features[:, :, t]
                    .unsqueeze(1)
                )

                pred_t, h01, h02 = model(
                    x_t,
                    h01,
                    h02,
                )

                masks.append(
                    pred_t.squeeze(1)
                )

            mask = torch.stack(
                masks,
                dim=2,
            )

            mic_spec = torch.stft(
                mic.unsqueeze(0),
                n_fft=n_fft,
                hop_length=hop,
                win_length=window_l,
                window=window,
                center=False,
                return_complex=True,
            )

            enhanced_spec = mask * mic_spec

            frames = torch.fft.irfft(
                enhanced_spec,
                n=n_fft,
                dim=1,
            )

            frames = (
                frames
                * window.view(1, -1, 1)
            )

            output_length = (
                n_fft
                + (T - 1) * hop
            )

            enhanced = torch.zeros(
                1,
                output_length,
                device=device,
            )

            norm = torch.zeros_like(
                enhanced
            )

            window_sq = window.square()

            for t in range(T):
                start = t * hop
                end = start + n_fft

                enhanced[
                    :, start:end
                ] += frames[:, :, t]

                norm[
                    :, start:end
                ] += window_sq

            enhanced = (
                enhanced
                / norm.clamp(min=1e-8)
            )

            length = min(
                enhanced.size(1),
                mic.numel(),
            )

            result = {
                "index": idx,
                "guid": guid,
                "scenario": scenario,
                "supervised": supervised,

                "enhanced":
                    enhanced[0, :length].cpu(),

                "mic":
                    mic[:length].cpu(),

                "lpb":
                    lpb[:length].cpu(),

                "target":
                    target[:length].cpu(),

                "mask":
                    mask[0].cpu(),
            }

            result["mixed"] = result["mic"]
            result["nearend"] = result["target"]

            outputs.append(result)

    return outputs
