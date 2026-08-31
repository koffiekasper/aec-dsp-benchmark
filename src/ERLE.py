import torch


def erle(mic, enhanced, eps=1e-12):
    mic = torch.as_tensor(
        mic,
        dtype=torch.float64
    ).flatten()

    enhanced = torch.as_tensor(
        enhanced,
        dtype=torch.float64
    ).flatten()

    length = min(
        mic.numel(),
        enhanced.numel()
    )

    mic = mic[:length]
    enhanced = enhanced[:length]

    mic_power = mic.square().mean()
    residual_power = enhanced.square().mean()

    return (
        10.0
        * torch.log10(
            (mic_power + eps)
            / (residual_power + eps)
        )
    ).item()


def framewise_erle(
    mic,
    enhanced,
    frame_size=1600,
    hop=800,
    eps=1e-12,
):
    mic = torch.as_tensor(
        mic,
        dtype=torch.float64
    ).flatten()

    enhanced = torch.as_tensor(
        enhanced,
        dtype=torch.float64
    ).flatten()

    length = min(
        mic.numel(),
        enhanced.numel()
    )

    mic = mic[:length]
    enhanced = enhanced[:length]

    values = []

    for start in range(
        0,
        length - frame_size + 1,
        hop
    ):
        end = start + frame_size

        mic_frame = mic[start:end]
        enhanced_frame = enhanced[start:end]

        mic_power = mic_frame.square().mean()
        residual_power = enhanced_frame.square().mean()

        score = (
            10.0
            * torch.log10(
                (mic_power + eps)
                / (residual_power + eps)
            )
        )

        values.append(score)

    if not values:
        return torch.empty(0)

    return torch.stack(values)