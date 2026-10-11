"""Shared physical velocity-command shaping utilities."""

from __future__ import annotations

import torch


def apply_velocity_deadzone(
    command: torch.Tensor,
    *,
    planar_deadzone_mps: float,
    yaw_deadzone_radps: float,
) -> torch.Tensor:
    """Return a command with independently applied planar and yaw dead zones.

    ``command`` may be a single ``(3,)`` command or a batch ending in a
    three-element velocity vector.  The input is never modified in place.
    """
    if command.shape[-1] != 3:
        raise ValueError("Velocity commands must end with [linear_x, linear_y, yaw_rate].")
    if planar_deadzone_mps < 0.0 or yaw_deadzone_radps < 0.0:
        raise ValueError("Velocity-command dead zones must be nonnegative.")

    filtered = command.clone()
    stationary_planar = torch.linalg.vector_norm(filtered[..., :2], dim=-1) < planar_deadzone_mps
    filtered[..., :2] = torch.where(
        stationary_planar.unsqueeze(-1), torch.zeros_like(filtered[..., :2]), filtered[..., :2]
    )
    stationary_yaw = torch.abs(filtered[..., 2]) < yaw_deadzone_radps
    filtered[..., 2] = torch.where(stationary_yaw, torch.zeros_like(filtered[..., 2]), filtered[..., 2])
    return filtered
