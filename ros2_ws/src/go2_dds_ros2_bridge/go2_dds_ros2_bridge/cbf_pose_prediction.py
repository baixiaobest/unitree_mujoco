"""Short-horizon propagation of a base pose using measured body motion."""

import math
import numpy as np


def propagate_base_pose(position, rotation, velocity_base, angular_rate_base, seconds):
    position = np.asarray(position, dtype=float)
    rotation = np.asarray(rotation, dtype=float)
    velocity = np.asarray(velocity_base, dtype=float)
    angular_rate = np.asarray(angular_rate_base, dtype=float)
    if (position.shape != (3,) or rotation.shape != (3, 3) or velocity.shape != (3,)
            or angular_rate.shape != (3,) or not math.isfinite(seconds) or seconds < 0
            or not all(np.isfinite(x).all() for x in (position, rotation, velocity, angular_rate))):
        raise ValueError("invalid_motion")
    angle = float(np.linalg.norm(angular_rate) * seconds)
    skew = np.array([[0, -angular_rate[2], angular_rate[1]],
                     [angular_rate[2], 0, -angular_rate[0]],
                     [-angular_rate[1], angular_rate[0], 0]], dtype=float)
    if angle < 1e-8:
        delta = np.eye(3) + seconds * skew
        midpoint = np.eye(3) + 0.5 * seconds * skew
    else:
        norm = float(np.linalg.norm(angular_rate))
        axis_skew = skew / norm
        square = axis_skew @ axis_skew
        delta = np.eye(3) + math.sin(angle) * axis_skew + (1 - math.cos(angle)) * square
        midpoint = np.eye(3) + math.sin(angle / 2) * axis_skew + (1 - math.cos(angle / 2)) * square
    return position + rotation @ midpoint @ velocity * seconds, rotation @ delta


def newest_sample_at(history, target_ns, max_age_ns):
    for stamp_ns, values in reversed(history):
        if stamp_ns <= target_ns:
            if target_ns - stamp_ns <= max_age_ns and np.isfinite(values).all():
                return values
            break
    raise ValueError("motion_sample_stale_or_missing")


def bounded_prediction_seconds(anchor_ns, target_ns, maximum_s):
    delta_ns = target_ns - anchor_ns
    if delta_ns <= 0 or delta_ns > int(maximum_s * 1e9):
        raise ValueError("prediction_horizon_exceeded_or_not_future")
    return delta_ns * 1e-9


def claim_pair_once(claimed, pair_key):
    if pair_key in claimed:
        return False
    claimed.add(pair_key)
    return True
