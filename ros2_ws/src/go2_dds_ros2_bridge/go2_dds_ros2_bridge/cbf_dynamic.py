"""Portable temporal_lidar_v4 projection and dynamic CBF point preparation."""

from __future__ import annotations

import numpy as np

CONTRACT = b"temporal_lidar_v4"
FRAMES = 4
RAYS = 256
WORLD_BINS = 256
FRONT_BINS = 128
MAX_DISTANCE_M = 20.0


def project_history(hit_xy, ray_state, evaluation_xy, evaluation_yaw):
    """Return (2,4,128) predictor input and newest reflection winner IDs.

    This is the deployment copy of lidar_velocity_predictor/src/projection.py.
    History is newest first; absent startup frames have state zero.
    """
    hits = np.asarray(hit_xy, dtype=np.float64)
    states = np.asarray(ray_state, dtype=np.uint8)
    xy = np.asarray(evaluation_xy, dtype=np.float64)
    if hits.shape != (FRAMES, RAYS, 2) or states.shape != (FRAMES, RAYS):
        raise ValueError("predictor_history_shape")
    if xy.shape != (2,) or not np.isfinite(xy).all() or not np.isfinite(evaluation_yaw):
        raise ValueError("invalid_evaluation_pose")
    if np.any(states > 2) or not np.isfinite(hits[states > 0]).all():
        raise ValueError("invalid_capture_rays")
    center = int((evaluation_yaw + np.pi) / (2 * np.pi) * WORLD_BINS) % WORLD_BINS
    arc = (center + np.arange(-FRONT_BINS // 2, FRONT_BINS // 2)) % WORLD_BINS
    output = np.zeros((2, FRAMES, FRONT_BINS), dtype=np.float32)
    output[0] = 1.0
    winner = np.full(FRONT_BINS, -1, dtype=np.int32)
    for age in range(FRAMES):
        delta = hits[age] - xy
        distance = np.minimum(np.linalg.norm(delta, axis=-1), MAX_DISTANCE_M)
        angle = np.arctan2(delta[:, 1], delta[:, 0])
        world_bin = np.floor((angle + np.pi) / (2 * np.pi) * WORLD_BINS).astype(np.int64) % WORLD_BINS
        for local_bin, global_bin in enumerate(arc):
            rays = np.flatnonzero((world_bin == global_bin) & (states[age] > 0))
            if rays.size == 0:
                continue
            output[1, age, local_bin] = 1.0
            reflections = rays[states[age, rays] == 2]
            if reflections.size:
                chosen = reflections[np.argmin(distance[reflections])]
                output[0, age, local_bin] = distance[chosen] / MAX_DISTANCE_M
                if age == 0:
                    winner[local_bin] = chosen
    return output, winner


def prepare_points(newest_hit_xy, winner, predicted_body, yaw, capture_age_s):
    """Rotate absolute velocities at evaluation yaw, then advance winning hits."""
    hits = np.asarray(newest_hit_xy, dtype=np.float64)
    indices = np.asarray(winner, dtype=np.int32)
    velocity = np.asarray(predicted_body, dtype=np.float64)
    if hits.shape != (RAYS, 2) or indices.shape != (FRONT_BINS,) or velocity.shape != (FRONT_BINS, 2):
        raise ValueError("dynamic_prediction_shape")
    if not np.isfinite(velocity).all() or not np.isfinite(capture_age_s) or capture_age_s < 0:
        raise ValueError("invalid_dynamic_prediction")
    c, s = np.cos(yaw), np.sin(yaw)
    velocity_world = np.stack((c * velocity[:, 0] - s * velocity[:, 1],
                               s * velocity[:, 0] + c * velocity[:, 1]), axis=1)
    reflected = indices >= 0
    points_world = np.zeros((FRONT_BINS, 2), dtype=np.float64)
    points_world[reflected] = hits[indices[reflected]] + velocity_world[reflected] * capture_age_s
    velocity_world[~reflected] = 0.0
    if not np.isfinite(points_world).all():
        raise ValueError("nonfinite_propagated_points")
    return points_world, velocity_world, reflected.astype(np.uint8)


def history_arrays(scans):
    """Convert newest-first CbfWorldScan messages to padded predictor arrays."""
    hits = np.zeros((FRAMES, RAYS, 2), dtype=np.float64)
    states = np.zeros((FRAMES, RAYS), dtype=np.uint8)
    for age, scan in enumerate(scans[:FRAMES]):
        hits[age] = np.asarray(scan.capture_points_xy_m, dtype=np.float64).reshape(RAYS, 2)
        states[age] = np.asarray(scan.ray_states, dtype=np.uint8)
    return hits, states
