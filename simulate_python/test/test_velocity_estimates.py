"""Tests for legacy and six-value estimator output handling."""

import pytest
import torch

from utils.velocity_estimates import split_velocity_estimate


def test_legacy_estimate_uses_linear_velocity_only():
    linear = torch.tensor([1.0, 2.0, 3.0])
    decoded_linear, decoded_angular = split_velocity_estimate(linear)
    torch.testing.assert_close(decoded_linear, linear)
    assert decoded_angular is None


def test_six_value_estimate_splits_angular_before_linear():
    estimate = torch.tensor([0.1, 0.2, 0.3, 1.0, 2.0, 3.0])
    linear, angular = split_velocity_estimate(estimate)
    torch.testing.assert_close(linear, torch.tensor([1.0, 2.0, 3.0]))
    torch.testing.assert_close(angular, torch.tensor([0.1, 0.2, 0.3]))


@pytest.mark.parametrize("estimate", [torch.zeros(4), torch.zeros(1, 3)])
def test_invalid_estimate_shape_fails(estimate):
    with pytest.raises(ValueError, match="velocity estimate"):
        split_velocity_estimate(estimate)
