"""Tests for deployment velocity-command shaping without simulator dependencies."""

import torch

from utils.velocity_commands import apply_velocity_deadzone


def test_velocity_deadzone_filters_planar_and_yaw_independently():
    command = torch.tensor([0.06, 0.06, 0.20])
    torch.testing.assert_close(
        apply_velocity_deadzone(command, planar_deadzone_mps=0.10, yaw_deadzone_radps=0.10),
        torch.tensor([0.0, 0.0, 0.20]),
    )

    command = torch.tensor([0.20, 0.0, 0.06])
    torch.testing.assert_close(
        apply_velocity_deadzone(command, planar_deadzone_mps=0.10, yaw_deadzone_radps=0.10),
        torch.tensor([0.20, 0.0, 0.0]),
    )


def test_velocity_deadzone_preserves_threshold_and_does_not_mutate_input():
    command = torch.tensor([0.10, 0.0, 0.10])
    filtered = apply_velocity_deadzone(command, planar_deadzone_mps=0.10, yaw_deadzone_radps=0.10)
    torch.testing.assert_close(filtered, command)
    torch.testing.assert_close(command, torch.tensor([0.10, 0.0, 0.10]))
