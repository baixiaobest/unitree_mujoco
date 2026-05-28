from __future__ import annotations

import math

import numpy as np


DEFAULT_LIDAR_TF_RPY_DEG = (192.0, -8.0, -60.0)
DEFAULT_LIDAR_TF_XYZ = (0.2929999828338623, 0.0, -0.06000000238418579)
DEFAULT_IMU_TF_XYZ = (-0.02557, 0.0, 0.04232)


def quaternion_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    half_roll = 0.5 * roll
    half_pitch = 0.5 * pitch
    half_yaw = 0.5 * yaw
    cr = math.cos(half_roll)
    sr = math.sin(half_roll)
    cp = math.cos(half_pitch)
    sp = math.sin(half_pitch)
    cy = math.cos(half_yaw)
    sy = math.sin(half_yaw)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def rotation_matrix_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def rotation_matrix_from_rpy_degrees(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    return rotation_matrix_from_rpy(
        math.radians(roll_deg),
        math.radians(pitch_deg),
        math.radians(yaw_deg),
    )


def normalize_vector(vector: np.ndarray, *, eps: float = 1e-9) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= eps:
        raise ValueError("Cannot normalize a near-zero vector.")
    return np.asarray(vector, dtype=np.float64) / norm


def rotation_matrix_from_quaternion_xyzw(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def roll_pitch_correction_from_gravity(gravity_vector: np.ndarray) -> tuple[float, float, np.ndarray]:
    gravity_direction = normalize_vector(gravity_vector)
    roll = math.atan2(-float(gravity_direction[1]), -float(gravity_direction[2]))
    roll_rotation = rotation_matrix_from_rpy(roll, 0.0, 0.0)
    gravity_after_roll = roll_rotation @ gravity_direction
    pitch = math.atan2(float(gravity_after_roll[0]), -float(gravity_after_roll[2]))
    correction_rotation = rotation_matrix_from_rpy(roll, pitch, 0.0)
    return roll, pitch, correction_rotation


def transform_matrix_from_xyz_rpy_degrees(
    x: float,
    y: float,
    z: float,
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_matrix_from_rpy_degrees(roll_deg, pitch_deg, yaw_deg)
    transform[:3, 3] = np.array((x, y, z), dtype=np.float64)
    return transform


def lidar_pose_in_imu_frame() -> tuple[np.ndarray, np.ndarray]:
    lidar_in_base = transform_matrix_from_xyz_rpy_degrees(
        DEFAULT_LIDAR_TF_XYZ[0],
        DEFAULT_LIDAR_TF_XYZ[1],
        DEFAULT_LIDAR_TF_XYZ[2],
        DEFAULT_LIDAR_TF_RPY_DEG[0],
        DEFAULT_LIDAR_TF_RPY_DEG[1],
        DEFAULT_LIDAR_TF_RPY_DEG[2],
    )
    imu_in_base = np.eye(4, dtype=np.float64)
    imu_in_base[:3, 3] = np.array(DEFAULT_IMU_TF_XYZ, dtype=np.float64)
    lidar_in_imu = np.linalg.inv(imu_in_base) @ lidar_in_base
    return lidar_in_imu[:3, 3].copy(), lidar_in_imu[:3, :3].copy()


def base_link_pose_in_imu_frame() -> tuple[np.ndarray, np.ndarray]:
    imu_in_base = np.eye(4, dtype=np.float64)
    imu_in_base[:3, 3] = np.array(DEFAULT_IMU_TF_XYZ, dtype=np.float64)
    base_in_imu = np.linalg.inv(imu_in_base)
    return base_in_imu[:3, 3].copy(), base_in_imu[:3, :3].copy()


def flatten_rotation_matrix(rotation: np.ndarray) -> list[float]:
    return [float(value) for value in rotation.reshape(-1)]


def quaternion_from_rotation_matrix(rotation: np.ndarray) -> tuple[float, float, float, float]:
    trace = float(rotation[0, 0] + rotation[1, 1] + rotation[2, 2])
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        qw = 0.25 * scale
        qx = (rotation[2, 1] - rotation[1, 2]) / scale
        qy = (rotation[0, 2] - rotation[2, 0]) / scale
        qz = (rotation[1, 0] - rotation[0, 1]) / scale
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
        qw = (rotation[2, 1] - rotation[1, 2]) / scale
        qx = 0.25 * scale
        qy = (rotation[0, 1] + rotation[1, 0]) / scale
        qz = (rotation[0, 2] + rotation[2, 0]) / scale
    elif rotation[1, 1] > rotation[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
        qw = (rotation[0, 2] - rotation[2, 0]) / scale
        qx = (rotation[0, 1] + rotation[1, 0]) / scale
        qy = 0.25 * scale
        qz = (rotation[1, 2] + rotation[2, 1]) / scale
    else:
        scale = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
        qw = (rotation[1, 0] - rotation[0, 1]) / scale
        qx = (rotation[0, 2] + rotation[2, 0]) / scale
        qy = (rotation[1, 2] + rotation[2, 1]) / scale
        qz = 0.25 * scale
    return float(qx), float(qy), float(qz), float(qw)


def rotation_vector_from_matrix(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    cosine = np.clip(0.5 * (trace - 1.0), -1.0, 1.0)
    angle = float(np.arccos(cosine))
    if angle < 1e-9:
        return np.zeros(3, dtype=np.float64)

    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-9:
        return np.zeros(3, dtype=np.float64)
    return axis * (angle / axis_norm)