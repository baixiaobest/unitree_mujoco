"""Decode legacy and current policy-estimator TorchScript outputs."""

import torch


def split_velocity_estimate(estimate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return body-frame (linear, angular) velocity from an estimator output.

    Legacy models output linear xyz. Current models output angular xyz followed by linear xyz,
    matching the sorted target paths in the estimator training dataset.
    """
    if estimate.ndim != 1:
        raise ValueError(f"Expected a one-dimensional velocity estimate, got shape {tuple(estimate.shape)}.")
    if estimate.numel() == 3:
        return estimate, None
    if estimate.numel() == 6:
        return estimate[3:6], estimate[0:3]
    raise ValueError(f"Expected 3 or 6 velocity estimate values, got {estimate.numel()}.")
