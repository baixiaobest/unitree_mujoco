"""ROS-independent geometry and bounded measured-state alignment for CBF."""
from bisect import bisect_left
import numpy as np
from .tf_utils import rotation_matrix_from_quaternion_xyzw


def bracket(history, stamp, max_gap_ns=100_000_000):
    times = [item[0] for item in history]
    i = bisect_left(times, stamp)
    if i < len(times) and times[i] == stamp:
        return history[i], history[i], 0.0
    if i == 0 or i == len(times):
        raise ValueError("state_not_bracketed")
    left, right = history[i - 1], history[i]
    if right[0] - left[0] > max_gap_ns:
        raise ValueError("state_interpolation_gap")
    return left, right, (stamp - left[0]) / (right[0] - left[0])


def slerp(a, b, fraction):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not np.isfinite(a).all() or not np.isfinite(b).all() or min(np.linalg.norm(a), np.linalg.norm(b)) < 1e-9:
        raise ValueError("invalid_quaternion")
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    dot = float(a @ b)
    if dot < 0:
        b, dot = -b, -dot
    if dot > .9995:
        q = a + fraction * (b - a)
        return q / np.linalg.norm(q)
    angle = np.arccos(np.clip(dot, -1, 1))
    return (np.sin((1-fraction)*angle)*a + np.sin(fraction*angle)*b) / np.sin(angle)


def pose_at(history, stamp, max_gap_ns=100_000_000):
    a, b, f = bracket(history, stamp, max_gap_ns)
    p = (1-f)*a[1] + f*b[1]
    q = slerp(a[2], b[2], f)
    return p, rotation_matrix_from_quaternion_xyzw(*q), (b[0]-a[0])*1e-9


def aligned_state(poses, velocities, world_rotation, world_translation, body_base_offset,
                  now_ns, max_age_ns=100_000_000, max_gap_ns=100_000_000):
    if not poses or not velocities:
        raise ValueError("state_unavailable")
    stamp = min(poses[-1][0], velocities[-1][0], now_ns)
    if now_ns - stamp > max_age_ns:
        raise ValueError("state_stale")
    p, rotation, gap = pose_at(poses, stamp, max_gap_ns)
    a, b, fraction = bracket(velocities, stamp, max_gap_ns)
    world_velocities = []
    for sample in (a, b):
        _, sample_rotation, sample_gap = pose_at(poses, sample[0], max_gap_ns)
        gap = max(gap, sample_gap)
        world_velocities.append(world_rotation @ sample_rotation @ sample[1])
    velocity = (1-fraction)*world_velocities[0] + fraction*world_velocities[1]
    rotation = world_rotation @ rotation
    position = world_rotation @ p + world_translation + rotation @ body_base_offset
    gap = max(gap, (b[0]-a[0])*1e-9)
    return stamp, position, np.arctan2(rotation[1, 0], rotation[0, 0]), velocity, gap


def world_points(points, position, rotation, world_rotation, world_translation, body_base_offset):
    """Full XYZ base->body->camera_init->world, before any XY projection."""
    return (np.asarray(points).reshape(-1, 3) + body_base_offset) @ rotation.T @ world_rotation.T + position @ world_rotation.T + world_translation
