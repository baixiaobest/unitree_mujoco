#!/usr/bin/env python3
r"""Inspect FAST-LIO diagnostics in a local Plotly viewer.

From source/unitree_mujoco/ros2_ws:
    source /opt/ros/humble/setup.bash
    source install/setup.bash
    python3 src/go2_dds_ros2_bridge/scripts/analyze_fast_lio_bag.py fast_lio_debug_bag/

Omit BAG_DIR to choose it in a file-picker window. The server listens only on
127.0.0.1; press Ctrl+C to stop it. A relative BAG_DIR uses the terminal's
current directory. No HTML report file is generated.

To record a bag while localization runs with match_debug:=true:
    ros2 bag record -o fast_lio_debug_bag \
      /Odometry /fast_lio/estimator_diagnostics /fast_lio/pose_information /fast_lio/match_debug \
      /fast_lio/velocity_fusion_debug /imu/data_fastlio_raw /odom /estimated_velocity /tf /tf_static

Press Ctrl+C to finish recording. The IMU topic is the Go2 body IMU.
"""

from __future__ import annotations

import argparse
import bisect
import io
import json
import math
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np


DIAGNOSTICS_TOPIC = "/fast_lio/estimator_diagnostics"
BODY_IMU_TOPIC = "/imu/data_fastlio_raw"
ODOMETRY_TOPIC = "/Odometry"
MATCH_DEBUG_TOPIC = "/fast_lio/match_debug"
POSE_INFORMATION_TOPIC = "/fast_lio/pose_information"
LEARNED_ODOMETRY_TOPIC = "/odom"
LEARNED_VELOCITY_TOPIC = "/estimated_velocity"
VELOCITY_FUSION_TOPIC = "/fast_lio/velocity_fusion_debug"


def stamp_seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def read_bag(bag, diagnostics_topic, imu_topic, odometry_topic, match_debug_topic):
    try:
        import rosbag2_py
    except ModuleNotFoundError as error:
        raise ImportError("rosbag2_py is unavailable. Source /opt/ros/humble/setup.bash "
                          "and this workspace's install/setup.bash, then use its Python interpreter.") from error
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    import yaml

    bag = bag.expanduser().resolve()
    metadata = bag / "metadata.yaml"
    if not metadata.is_file():
        raise ValueError(f"No metadata.yaml in {bag}; relative paths use the terminal's current directory")
    info = yaml.safe_load(metadata.read_text())["rosbag2_bagfile_information"]
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id=info["storage_identifier"]),
        rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    topic_types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    expected = {diagnostics_topic: "fast_lio/msg/EstimatorDiagnostics",
                imu_topic: "sensor_msgs/msg/Imu",
                odometry_topic: "nav_msgs/msg/Odometry",
                match_debug_topic: "fast_lio/msg/MatchDebug",
                POSE_INFORMATION_TOPIC: "fast_lio/msg/PoseInformation",
                LEARNED_ODOMETRY_TOPIC: "nav_msgs/msg/Odometry",
                LEARNED_VELOCITY_TOPIC: "geometry_msgs/msg/TwistStamped"}
    expected[VELOCITY_FUSION_TOPIC] = "fast_lio/msg/VelocityFusionDebug"
    if diagnostics_topic not in topic_types:
        raise ValueError(f"Missing {diagnostics_topic}; available topics: {', '.join(sorted(topic_types))}")
    for topic, type_name in expected.items():
        if topic in topic_types and topic_types[topic] != type_name:
            raise ValueError(f"{topic} has unexpected type {topic_types[topic]}")
    selected = [name for name in (*expected, "/tf", "/tf_static") if name in topic_types]
    classes = {name: get_message(topic_types[name]) for name in selected}
    legacy_diagnostics_class = get_message("fast_lio/msg/EstimatorDiagnosticsLegacy")
    diagnostics_class = classes[diagnostics_topic]
    legacy_match_class = get_message("fast_lio/msg/MatchDebugLegacy") if match_debug_topic in classes else None
    match_class = classes.get(match_debug_topic)
    messages = {name: [] for name in selected}
    while reader.has_next():
        topic, serialized, _ = reader.read_next()
        if topic in classes:
            if topic == diagnostics_topic:
                try:
                    message = deserialize_message(serialized, diagnostics_class)
                except Exception:
                    if diagnostics_class == legacy_diagnostics_class:
                        raise
                    diagnostics_class = legacy_diagnostics_class
                    message = deserialize_message(serialized, legacy_diagnostics_class)
                messages[topic].append(message)
            elif topic == match_debug_topic:
                try:
                    message = deserialize_message(serialized, match_class)
                except Exception:
                    if match_class == legacy_match_class:
                        raise
                    match_class = legacy_match_class
                    message = deserialize_message(serialized, legacy_match_class)
                messages[topic].append(message)
            else:
                messages[topic].append(deserialize_message(serialized, classes[topic]))
    for name, items in messages.items():
        if name not in ("/tf", "/tf_static"):
            items.sort(key=lambda item: stamp_seconds(item.header.stamp))
    return (messages[diagnostics_topic], messages.get(imu_topic, []),
            messages.get(odometry_topic, []), messages.get(match_debug_topic, []),
            messages.get("/tf", []) + messages.get("/tf_static", []),
            messages.get(POSE_INFORMATION_TOPIC, []),
            messages.get(LEARNED_ODOMETRY_TOPIC, []),
            messages.get(LEARNED_VELOCITY_TOPIC, []),
            messages.get(VELOCITY_FUSION_TOPIC, []))


def read_pcd_xyz(stream):
    """Read finite XYZ points from a standard ASCII or binary PCD stream."""
    header = {}
    while True:
        line = stream.readline()
        if not line:
            raise ValueError("Invalid PCD header")
        parts = line.decode("ascii", errors="replace").strip().split()
        if parts and not parts[0].startswith("#"):
            header[parts[0].upper()] = parts[1:]
        if parts and parts[0].upper() == "DATA":
            break
    fields = header["FIELDS"]
    sizes = [int(value) for value in header["SIZE"]]
    kinds = header["TYPE"]
    counts = [int(value) for value in header.get("COUNT", ["1"] * len(fields))]
    count = int(header["POINTS"][0]) if "POINTS" in header else int(header["WIDTH"][0]) * int(header["HEIGHT"][0])
    if not all(axis in fields for axis in "xyz"):
        raise ValueError("PCD requires x, y, z fields")
    if header["DATA"][0] == "ascii":
        columns = [sum(counts[:fields.index(axis)]) for axis in "xyz"]
        points = np.loadtxt(stream, usecols=columns, ndmin=2, dtype=np.float32)
    elif header["DATA"][0] == "binary":
        dtype_fields = []
        codes = {("F", 4): "<f4", ("F", 8): "<f8", ("I", 1): "i1", ("I", 2): "<i2",
                 ("I", 4): "<i4", ("U", 1): "u1", ("U", 2): "<u2", ("U", 4): "<u4"}
        for name, size, kind, n in zip(fields, sizes, kinds, counts):
            code = codes.get((kind, size))
            if code is None:
                raise ValueError(f"Unsupported PCD field {name}: {kind}{size}")
            dtype_fields.append((name, code, (n,)) if n != 1 else (name, code))
        dtype = np.dtype(dtype_fields)
        records = np.frombuffer(stream.read(count * dtype.itemsize), dtype=dtype, count=count)
        points = np.column_stack([records[axis] for axis in "xyz"]).astype(np.float32)
    else:
        raise ValueError("PCD must use ASCII or binary encoding")
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        raise ValueError("PCD has no finite XYZ points")
    return points


def quaternion_to_rpy(q):
    norm = math.sqrt(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z)
    if norm < 1e-8:
        return [None, None, None]
    w, x, y, z = (v / norm for v in (q.w, q.x, q.y, q.z))
    return [math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
            math.asin(max(-1, min(1, 2*(w*y-z*x)))),
            math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))]


def rotation_matrix(q):
    w, x, y, z = q.w, q.x, q.y, q.z
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    if norm < 1e-8:
        raise ValueError("Invalid transform quaternion")
    w, x, y, z = (v / norm for v in (w, x, y, z))
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def rotation_change(previous_xyzw, predicted_xyzw):
    """Shortest body-frame rotation vector from the previous pose to the prediction."""
    previous = np.asarray(previous_xyzw, dtype=np.float64)
    predicted = np.asarray(predicted_xyzw, dtype=np.float64)
    if not np.isfinite(previous).all() or not np.isfinite(predicted).all():
        return None
    previous_norm, predicted_norm = np.linalg.norm(previous), np.linalg.norm(predicted)
    if previous_norm < 1e-8 or predicted_norm < 1e-8:
        return None
    x, y, z, w = previous / previous_norm
    a, b, c, d = predicted / predicted_norm
    # Conjugate(previous) * predicted, with quaternions stored xyzw.
    vector = np.array([w*a - x*d - y*c + z*b,
                       w*b + x*c - y*d - z*a,
                       w*c - x*b + y*a - z*d])
    scalar = w*d + x*a + y*b + z*c
    if scalar < 0:
        vector, scalar = -vector, -scalar
    length = np.linalg.norm(vector)
    return vector * (2 * math.atan2(length, scalar) / length) if length > 1e-12 else 2 * vector


def transform_points(points, transform):
    return points @ rotation_matrix(transform.rotation).T + np.array(
        [transform.translation.x, transform.translation.y, transform.translation.z])


def point_array(points):
    return np.array([[p.x, p.y, p.z] for p in points], dtype=np.float32).reshape(-1, 3)


def finite(value):
    return float(value) if math.isfinite(value) else None


def translation_constraints(scan):
    normals = point_array(getattr(scan, "match_normals", []))
    if not scan.update_valid or not len(normals):
        return None
    if len(normals) != len(scan.matched_indices):
        raise ValueError("Match debug has a different number of normals and matched returns")
    lengths = np.linalg.norm(normals, axis=1)
    if not np.isfinite(normals).all() or np.any(lengths < 1e-8):
        raise ValueError("Match debug contains invalid plane normals")
    normals = normals / lengths[:, None]
    values, vectors = np.linalg.eigh(normals.T @ normals)
    values = np.maximum(values, 0.0)
    return values, vectors


def pose_information_matrix(message):
    if not message.update_valid:
        return None
    values = np.asarray(message.upper_triangle, dtype=np.float64)
    if values.shape != (21,) or not np.isfinite(values).all():
        raise ValueError("Pose information has invalid matrix entries")
    matrix = np.zeros((6, 6), dtype=np.float64)
    rows, columns = np.triu_indices(6)
    matrix[rows, columns] = values
    matrix[columns, rows] = values
    return matrix


def odometry_series(messages):
    if not messages:
        return None
    times = np.array([stamp_seconds(msg.header.stamp) for msg in messages], dtype=np.float64)
    positions = np.array([[getattr(msg.pose.pose.position, axis) for axis in "xyz"]
                          for msg in messages], dtype=np.float64)
    quaternions = np.array([[getattr(msg.pose.pose.orientation, axis) for axis in "wxyz"]
                            for msg in messages], dtype=np.float64)
    norms = np.linalg.norm(quaternions, axis=1)
    if not np.isfinite(positions).all() or not np.isfinite(quaternions).all() or np.any(norms < 1e-8):
        raise ValueError("Odometry contains invalid poses")
    return times, positions, quaternions / norms[:, None]


def quaternion_matrix(quaternion):
    w, x, y, z = quaternion
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def interpolated_pose(series, stamp):
    if series is None:
        return None
    times, positions, quaternions = series
    if stamp < times[0] or stamp > times[-1]:
        return None
    right = int(np.searchsorted(times, stamp, side="left"))
    if right == 0 or times[right] == stamp:
        return positions[right].copy(), quaternion_matrix(quaternions[right])
    left = right - 1
    fraction = (stamp-times[left]) / (times[right]-times[left])
    position = positions[left] + fraction*(positions[right]-positions[left])
    q0, q1 = quaternions[left], quaternions[right].copy()
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        quaternion = q0 + fraction*(q1-q0)
    else:
        angle = math.acos(np.clip(dot, -1, 1))
        quaternion = (math.sin((1-fraction)*angle)*q0 + math.sin(fraction*angle)*q1) / math.sin(angle)
    quaternion /= np.linalg.norm(quaternion)
    return position, quaternion_matrix(quaternion)


def aligned_paths(fast_series, learned_series, stamp, map_transform, max_points=1200):
    fast_start = interpolated_pose(fast_series, stamp)
    if fast_start is None:
        return None, None, None
    fast_position, fast_rotation = fast_start
    map_rotation = rotation_matrix(map_transform.rotation)
    map_translation = np.array([getattr(map_transform.translation, axis) for axis in "xyz"])

    def sample_path(series, start_position, alignment):
        times, positions, _ = series
        end = int(np.searchsorted(times, min(times[-1], fast_series[0][-1]), side="right"))
        start = int(np.searchsorted(times, stamp, side="right"))
        indices = np.arange(start, end)
        if len(indices) > max_points-1:
            indices = indices[np.unique(np.linspace(0, len(indices)-1, max_points-1, dtype=int))]
        world = fast_position + (positions[indices]-start_position) @ alignment.T
        world = np.vstack((fast_position, world))
        return np.round(world @ map_rotation.T + map_translation, 4).tolist()

    fast_path = sample_path(fast_series, fast_position, np.eye(3))
    learned_path = None
    learned_start = interpolated_pose(learned_series, stamp)
    if learned_start is not None:
        learned_position, learned_rotation = learned_start
        learned_path = sample_path(learned_series, learned_position, fast_rotation @ learned_rotation.T)
    map_pose = fast_position @ map_rotation.T + map_translation
    return fast_path, learned_path, (map_pose, map_rotation @ fast_rotation)


class ViewerData:
    def __init__(self, diagnostics, imu, odometry, matches, tf_messages, map_pcd, map_point_limit,
                 pose_information=(), learned_odometry=(), learned_velocity=(), velocity_fusion=()):
        if not diagnostics:
            raise ValueError("Diagnostics topic contains no messages")
        if map_point_limit < 1:
            raise ValueError("--map-point-limit must be positive")
        wrong_frames = sorted({item.header.frame_id for item in imu if item.header.frame_id != "body"})
        if wrong_frames:
            raise ValueError(f"Body IMU topic contains unexpected frame IDs: {wrong_frames}")
        self.first = stamp_seconds(diagnostics[0].header.stamp)
        self.diagnostics = diagnostics
        self.imu = imu
        self.odometry = odometry
        self.odom_times = [stamp_seconds(item.header.stamp) for item in odometry]
        self.pose_information = pose_information
        self.learned_odometry = learned_odometry
        self.learned_velocity = learned_velocity
        self.velocity_fusion = velocity_fusion
        self.fast_series = odometry_series(odometry)
        self.learned_series = odometry_series(learned_odometry)
        self.information_times = [stamp_seconds(msg.header.stamp) for msg in pose_information]
        self.map_point_limit = map_point_limit
        self.map_points = np.empty((0, 3), dtype=np.float32)
        if map_pcd is not None:
            with map_pcd.expanduser().open("rb") as stream:
                self.set_map(stream)
        self.scans = []
        self.constraints = []
        self.skipped_scans = 0
        if matches:
            frames = {item.header.frame_id for item in matches}
            if len(frames) != 1:
                raise ValueError(f"Match debug changes world frame: {sorted(frames)}")
            frame = frames.pop().lstrip("/")
            transforms = []
            for message in tf_messages:
                for item in message.transforms:
                    if item.header.frame_id.lstrip("/") == "map" and item.child_frame_id.lstrip("/") == frame:
                        transforms.append((stamp_seconds(item.header.stamp), item.transform))
            transforms.sort(key=lambda item: item[0])
            if not transforms:
                raise ValueError(f"No map-to-{frame} transform in recorded /tf")
            tf_times = [item[0] for item in transforms]
            for scan in matches:
                stamp = stamp_seconds(scan.header.stamp)
                index = bisect.bisect_right(tf_times, stamp) - 1
                if index < 0:
                    self.skipped_scans += 1
                    continue
                self.scans.append((scan, transforms[index][1]))
                self.constraints.append(translation_constraints(scan))
        self.overview = self._make_overview()

    def set_map(self, stream):
        points = read_pcd_xyz(stream)
        step = max(1, math.ceil(len(points) / self.map_point_limit))
        self.map_points = points[::step].copy()
        return len(self.map_points)

    def map_payload(self):
        return {"points": np.round(self.map_points, 3).tolist(), "count": len(self.map_points)}

    def scan_index(self):
        return {"times": [round(stamp_seconds(scan.header.stamp) - self.first, 3)
                          for scan, _ in self.scans],
                "skipped": self.skipped_scans}

    def scan_payload(self, index):
        scan, transform = self.scans[index]
        constraint = self.constraints[index]
        count = len(scan.matched_indices)
        if len(scan.map_neighbors) != 5*count or len(scan.signed_residuals) != count:
            raise ValueError(f"Malformed match debug scan {index}: expected five map neighbors per match")
        if any(i >= len(scan.downsampled_returns) for i in scan.matched_indices):
            raise ValueError(f"Malformed match debug scan {index}: matched index out of range")
        points = transform_points(point_array(scan.downsampled_returns), transform)
        indices = np.asarray(scan.matched_indices, dtype=np.int64)
        mask = np.ones(len(points), dtype=bool)
        mask[indices] = False
        stamp = stamp_seconds(scan.header.stamp)
        fast_path, learned_path, anchor = aligned_paths(
            self.fast_series, self.learned_series, stamp, transform)
        pose = tip = rotation_axes = None
        if anchor is not None:
            pose_xyz, map_imu_rotation = anchor
            pose = np.round(pose_xyz, 4).tolist()
            tip = np.round(pose_xyz + map_imu_rotation @ np.array([0.4, 0., 0.]), 4).tolist()
            if self.information_times:
                nearest = bisect.bisect_left(self.information_times, stamp)
                choices = [i for i in (nearest-1, nearest) if 0 <= i < len(self.information_times)]
                if choices:
                    chosen = min(choices, key=lambda i: abs(self.information_times[i]-stamp))
                    if abs(self.information_times[chosen]-stamp) < 0.2:
                        matrix = pose_information_matrix(self.pose_information[chosen])
                        if matrix is not None:
                            values, vectors = np.linalg.eigh(matrix[3:6, 3:6])
                            rotation_axes = {"values": np.maximum(values, 0).round(5).tolist(),
                                             "vectors": np.round(map_imu_rotation @ vectors, 6).T.tolist()}
        axes = None
        if constraint is not None:
            values, vectors = constraint
            axes = {"values": np.round(values, 5).tolist(),
                    "vectors": np.round(rotation_matrix(transform.rotation) @ vectors, 6).T.tolist()}
        return {"time": round(stamp-self.first, 3),
                "matched": np.round(points[indices], 4).tolist(),
                "unmatched": np.round(points[mask], 4).tolist(),
                "pose": pose, "pose_tip": tip, "constraints": axes,
                "rotation_constraints": rotation_axes,
                "fast_path": fast_path, "learned_path": learned_path,
                "valid": bool(scan.update_valid)}

    def match_payload(self, scan_index, match_index):
        scan, transform = self.scans[scan_index]
        if not 0 <= match_index < len(scan.matched_indices):
            raise ValueError("Matched return index out of range")
        point_index = scan.matched_indices[match_index]
        if point_index >= len(scan.downsampled_returns):
            raise ValueError("Matched return points outside the downsampled scan")
        start = 5 * match_index
        neighbors = scan.map_neighbors[start:start+5]
        if len(neighbors) != 5:
            raise ValueError("Matched return does not have five map neighbors")
        point = transform_points(point_array([scan.downsampled_returns[point_index]]), transform)[0]
        map_points = transform_points(point_array(neighbors), transform)
        return {"scan_index": scan_index, "match_index": match_index,
                "point": np.round(point, 4).tolist(),
                "neighbors": np.round(map_points, 4).tolist(),
                "residual": finite(scan.signed_residuals[match_index])}

    def _make_overview(self):
        diagnostics, imu, odometry = self.diagnostics, self.imu, self.odometry
        x = [stamp_seconds(item.header.stamp)-self.first for item in diagnostics]
        plots = []

        def add(key, title, unit, traces):
            plots.append({"key": key, "title": title, "unit": unit,
                          "traces": [{"name": name, "x": times, "y": [finite(v) if v is not None else None for v in values]}
                                     for name, times, values in traces]})

        for field, title, unit in (("position_stddev", "Position standard deviation", "m"),
                                   ("rotation_stddev", "Rotation standard deviation", "rad")):
            add(field, title, unit, [(axis, x, [getattr(item, field)[i] for item in diagnostics])
                                    for i, axis in enumerate("xyz")])
        if odometry:
            ox = [stamp_seconds(item.header.stamp)-self.first for item in odometry]
            add("position", "Estimated position", "m",
                [(axis, ox, [getattr(item.pose.pose.position, axis) for item in odometry]) for axis in "xyz"])
            angles = [quaternion_to_rpy(item.pose.pose.orientation) for item in odometry]
            add("rotation", "Estimated rotation (roll, pitch, yaw)", "rad",
                [(axis, ox, [a[i] for a in angles]) for i, axis in enumerate(("roll", "pitch", "yaw"))])
            by_stamp = {(item.header.stamp.sec, item.header.stamp.nanosec): item for item in odometry}
            position_changes = [None] * len(diagnostics)
            rotation_changes = [None] * len(diagnostics)
            for index in range(1, len(diagnostics)):
                item, previous = diagnostics[index], diagnostics[index-1]
                current_pose = by_stamp.get((item.header.stamp.sec, item.header.stamp.nanosec))
                previous_pose = by_stamp.get((previous.header.stamp.sec, previous.header.stamp.nanosec))
                if current_pose is None or previous_pose is None:
                    continue
                correction = np.asarray(item.lidar_position_correction, dtype=np.float64)
                if item.update_valid and np.isfinite(correction).all():
                    current_xyz = np.array([getattr(current_pose.pose.pose.position, axis) for axis in "xyz"])
                    previous_xyz = np.array([getattr(previous_pose.pose.pose.position, axis) for axis in "xyz"])
                    position_changes[index] = current_xyz - correction - previous_xyz
                predicted_quaternion = getattr(item, "predicted_orientation_xyzw", None)
                if predicted_quaternion is not None:
                    previous_quaternion = previous_pose.pose.pose.orientation
                    rotation_changes[index] = rotation_change(
                        [getattr(previous_quaternion, axis) for axis in "xyzw"], predicted_quaternion)
            if any(change is not None for change in position_changes):
                prediction_source = "IMU + learned velocity" if any(item.status == 0 for item in self.velocity_fusion) else "IMU"
                add("predicted_position_change", f"{prediction_source}-predicted position change", "m / scan",
                    [(axis, x, [float(change[i]) if change is not None else None
                                for change in position_changes]) for i, axis in enumerate("xyz")] +
                    [("magnitude", x, [float(np.linalg.norm(change)) if change is not None else None
                                       for change in position_changes])])
            if any(change is not None for change in rotation_changes):
                prediction_source = "IMU + learned velocity" if any(item.status == 0 for item in self.velocity_fusion) else "IMU"
                add("predicted_rotation_change", f"{prediction_source}-predicted rotation change (body frame)", "rad / scan",
                    [(axis, x, [float(change[i]) if change is not None else None
                                for change in rotation_changes]) for i, axis in enumerate("xyz")] +
                    [("magnitude", x, [float(np.linalg.norm(change)) if change is not None else None
                                       for change in rotation_changes])])
        if self.learned_odometry:
            lx = [stamp_seconds(item.header.stamp)-self.first for item in self.learned_odometry]
            add("learned_position", "Learned-velocity odometry position", "m",
                [(axis, lx, [getattr(item.pose.pose.position, axis) for item in self.learned_odometry])
                 for axis in "xyz"])
            add("learned_angular", "Body gyroscope in learned odometry", "rad/s",
                [(axis, lx, [getattr(item.twist.twist.angular, axis) for item in self.learned_odometry])
                 for axis in "xyz"])
        velocity_messages = self.learned_velocity or self.learned_odometry
        if velocity_messages:
            vx = [stamp_seconds(item.header.stamp)-self.first for item in velocity_messages]
            add("learned_velocity", "Learned body-frame linear velocity", "m/s",
                [(axis, vx, [getattr((item.twist if self.learned_velocity else item.twist.twist).linear, axis)
                              for item in velocity_messages]) for axis in "xyz"])
        add("matches", "Match availability", "points",
            [("downsampled", x, [item.downsampled_points for item in diagnostics]),
             ("matched", x, [item.matched_points for item in diagnostics])])
        add("match_ratio", "Match ratio", "fraction",
            [("matched / downsampled", x, [item.matched_points/item.downsampled_points
              if item.update_valid and item.downsampled_points else None for item in diagnostics])])
        if any(item is not None for item in self.constraints):
            cx = [stamp_seconds(scan.header.stamp)-self.first for scan, _ in self.scans]
            for normalized, key, title, unit in (
                    (False, "constraint_raw", "Translation constraint eigenvalues (raw)", "matches"),
                    (True, "constraint_normalized", "Translation constraint eigenvalues (per match)", "fraction")):
                add(key, title, unit,
                    [(name, cx, [float(item[0][i] / (sum(item[0]) if normalized else 1))
                                 if item is not None and (not normalized or sum(item[0]) > 0) else None
                                 for item in self.constraints])
                     for i, name in enumerate(("weakest", "middle", "strongest"))])
        if self.pose_information:
            px = [stamp_seconds(item.header.stamp)-self.first for item in self.pose_information]
            matrices = [pose_information_matrix(item) for item in self.pose_information]
            eigens = [np.maximum(np.linalg.eigvalsh(m[3:6, 3:6]), 0) if m is not None else None
                      for m in matrices]
            for per_match in (False, True):
                add("rotation_information" + ("_per_match" if per_match else ""),
                    "Rotation constraint eigenvalues" + (" (per match)" if per_match else ""), "1/rad²",
                    [(name, px, [float(v[i] / item.matched_points if per_match else v[i]) if v is not None and
                       (not per_match or item.matched_points) else None
                       for v, item in zip(eigens, self.pose_information)])
                     for i, name in enumerate(("weakest", "middle", "strongest"))])
            # With a 1 m lever arm, rotation is expressed as equivalent point displacement.
            coupled = [np.maximum(np.linalg.eigvalsh(m), 0) if m is not None else None for m in matrices]
            add("pose_information", "Coupled pose constraint eigenvalues (1 m rotation lever arm)", "1/m²",
                [(f"mode {i+1}", px, [float(v[i]) if v is not None else None for v in coupled])
                 for i in range(6)])
        add("residual", "Mean absolute point-to-plane residual", "m",
            [("residual", x, [item.mean_abs_residual if item.update_valid else None for item in diagnostics])])
        add("position_correction", "LiDAR position correction", "m",
            [(axis, x, [item.lidar_position_correction[i] if item.update_valid else None for item in diagnostics])
             for i, axis in enumerate("xyz")] +
            [("magnitude", x, [math.sqrt(sum(v*v for v in item.lidar_position_correction))
              if item.update_valid else None for item in diagnostics])])
        add("rotation_correction", "LiDAR rotation correction", "rad",
            [("angle", x, [item.lidar_rotation_correction if item.update_valid else None for item in diagnostics])])
        add("update_valid", "LiDAR update applied", "0 / 1",
            [("valid", x, [int(item.update_valid) for item in diagnostics])])
        if imu:
            ix = [stamp_seconds(item.header.stamp)-self.first for item in imu]
            for key, field, title, unit in (("accel", "linear_acceleration", "Body accelerometer", "m/s²"),
                                            ("gyro", "angular_velocity", "Body gyroscope", "rad/s")):
                add(key, title, unit,
                    [(axis, ix, [getattr(getattr(item, field), axis) for item in imu]) for axis in "xyz"] +
                    [("magnitude", ix, [math.sqrt(sum(getattr(getattr(item, field), axis)**2 for axis in "xyz"))
                                       for item in imu])])
        if self.velocity_fusion:
            fusion = self.velocity_fusion
            fx = [stamp_seconds(item.header.stamp)-self.first for item in fusion]
            accepted = [item.status == 0 for item in fusion]
            add("fusion_status", "Learned velocity fusion status", "0 accepted, 1 late, 2 invalid, 3 numerical",
                [("status", fx, [item.status for item in fusion])])
            late_count = np.cumsum([item.status == 1 for item in fusion])
            add("fusion_late_rate", "Cumulative late-discard rate", "%",
                [("late", fx, [100*count/(i+1) for i, count in enumerate(late_count)])])
            add("fusion_delay", "Learned velocity delivery and processing age", "ms",
                [("observation to arrival", fx, [1000*(stamp_seconds(item.arrival_stamp)-stamp_seconds(item.header.stamp))
                                              for item in fusion]),
                 ("observation to processing", fx, [1000*(stamp_seconds(item.processing_stamp)-stamp_seconds(item.header.stamp))
                                                 for item in fusion])])
            for key, title, unit, getter in (
                ("fusion_velocity", "Learned and predicted base velocity", "m/s",
                 lambda item: (item.learned_velocity, item.predicted_velocity)),
                ("fusion_residual", "Learned velocity innovation", "m/s",
                 lambda item: (item.residual,)),
                ("fusion_velocity_change", "IMU-frame velocity before and after fusion", "m/s",
                 lambda item: (item.velocity_before, item.velocity_after)),
                ("fusion_bias", "Estimated IMU biases after velocity fusion", "sensor units",
                 lambda item: (item.gyro_bias, item.accel_bias))):
                labels = {"fusion_velocity": ("learned", "predicted"),
                          "fusion_residual": ("innovation",),
                          "fusion_velocity_change": ("before", "after"),
                          "fusion_bias": ("gyro", "accelerometer")}[key]
                add(key, title, unit,
                    [(f"{label} {axis}", fx,
                      [float(getter(item)[group][i]) if accepted[index] else None
                       for index, item in enumerate(fusion)])
                     for group, label in enumerate(labels) for i, axis in enumerate("xyz")])
            add("fusion_nis", "Learned velocity normalized innovation squared", "χ² (3 DoF)",
                [("NIS", fx, [item.nis if valid else None for item, valid in zip(fusion, accepted)])])
            add("fusion_correction", "Velocity fusion state correction", "mixed units",
                [(name, fx, [float(np.linalg.norm(item.state_correction[start:start+count]))
                             if valid else None for item, valid in zip(fusion, accepted)])
                 for name, start, count in (("position m", 0, 3), ("rotation rad", 3, 3),
                                            ("velocity m/s", 12, 3), ("gyro bias rad/s", 15, 3),
                                            ("accel bias m/s²", 18, 3))])
            add("fusion_gain", "Velocity fusion Kalman gain by state component", "gain norm",
                [(name, fx, [float(np.linalg.norm(np.asarray(item.kalman_gain).reshape(23, 3)[start:start+count]))
                             if valid else None for item, valid in zip(fusion, accepted)])
                 for name, start, count in (("position", 0, 3), ("rotation", 3, 3),
                                            ("velocity", 12, 3), ("gyro bias", 15, 3),
                                            ("accel bias", 18, 3))])
            add("fusion_covariance", "Velocity covariance diagonal after fusion", "(m/s)²",
                [(axis, fx, [item.covariance_diagonal[12+i] if valid else None
                             for item, valid in zip(fusion, accepted)]) for i, axis in enumerate("xyz")])
        groups = {"pose": {"position", "rotation", "position_stddev", "rotation_stddev",
                           "position_correction", "rotation_correction",
                           "predicted_position_change", "predicted_rotation_change"},
                  "matching": {"matches", "match_ratio", "residual", "update_valid"},
                  "imu": {"accel", "gyro"},
                  "learned": {"learned_position", "learned_velocity", "learned_angular"},
                  "fusion": {"fusion_status", "fusion_late_rate", "fusion_delay", "fusion_velocity",
                             "fusion_residual", "fusion_velocity_change", "fusion_bias", "fusion_nis",
                             "fusion_correction", "fusion_gain", "fusion_covariance"}}
        for plot in plots:
            plot["category"] = next((group for group, keys in groups.items() if plot["key"] in keys),
                                    "constraints")
        return {"summary": f"{len(diagnostics)} scans; {sum(bool(item.update_valid) for item in diagnostics)} valid updates; "
                           f"{len(odometry)} poses; {len(self.learned_odometry)} learned odometry poses; "
                           f"{len(imu)} body IMU samples; {len(self.velocity_fusion)} velocity fusion events; "
                           f"{len(self.scans)} 3D scans.",
                "plots": plots, "has_matches": bool(self.scans),
                "has_learned_odometry": bool(self.learned_odometry),
                "has_pose_information": any(item.update_valid for item in self.pose_information),
                "has_translation_constraints": any(item is not None for item in self.constraints)}


VIEWER_HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>FAST-LIO analyzer</title>
<script src="/plotly.js"></script><style>
body{font-family:system-ui,sans-serif;max-width:1250px;margin:1.5rem auto;padding:0 1rem;color:#182333}
nav{display:flex;gap:.5rem;border-bottom:1px solid #dae1e8;margin:1rem 0;overflow-x:auto}
nav button{white-space:nowrap}
nav button{border:1px solid #dae1e8;background:#eef3f8;padding:.65rem 1rem;cursor:pointer;border-radius:.5rem .5rem 0 0}
nav button.active{background:white;font-weight:600}
.controls,.row{display:flex;flex-wrap:wrap;align-items:center;gap:.7rem 1.2rem;margin:1rem 0}
.controls{padding:1rem;background:#eef3f8;border-radius:.6rem}
section{border:1px solid #dae1e8;border-radius:.6rem;margin:1rem 0;padding:.5rem}
section h2{font-size:1.1rem;margin:.25rem .5rem .6rem}
section label{margin:0 .5rem}section input[type=number]{width:5em}
#scan-slider{width:min(50vw,500px)}#match-plot{height:700px}
[hidden]{display:none!important}
</style></head><body>
<h1>FAST-LIO estimator diagnostics</h1><p id="summary">Loading bag…</p>
<nav><button class="series-tab active" data-category="pose">Pose</button>
<button class="series-tab" data-category="matching">LiDAR matching</button>
<button class="series-tab" data-category="constraints">Constraints</button>
<button class="series-tab" data-category="imu">Body IMU</button>
<button class="series-tab" data-category="learned">Learned odometry</button>
<button class="series-tab" data-category="fusion">Velocity fusion</button>
<button id="tab-match" hidden>3D scan match</button></nav>
<main id="series-panel"><div id="series-status" aria-live="polite"></div><div id="plots"></div>
<p>Each plot has its own trailing smoothing window. Set it to 1 for raw data.
These signals describe estimator consistency; absolute accuracy requires ground truth.</p></main>
<main id="match-panel" hidden>
<h2>3D scan match</h2>
<div class="row"><label>Reference PCD <input type="file" id="pcd-file" accept=".pcd"></label><span id="map-status"></span></div>
<form class="row" id="pcd-path-form"><label>Or enter PCD path
<input type="text" id="pcd-path" value="~/.local/share/go2_fast_lio/maps/map.pcd" size="52"></label>
<button type="submit">Load path</button></form>
<div class="row"><label>Scan <input type="range" id="scan-slider" min="0" value="0"></label>
<button type="button" id="scan-prev" aria-label="Previous scan" title="Previous scan">←</button>
<button type="button" id="scan-next" aria-label="Next scan" title="Next scan">→</button>
<select id="scan-select" aria-label="Select scan timestamp"></select><span id="scan-summary"></span></div>
<div class="controls">
<label><input type="checkbox" id="show-map" checked>Reference map</label>
<label><input type="checkbox" id="show-matched" checked>Matched returns</label>
<label><input type="checkbox" id="show-unmatched" checked>Unmatched returns</label>
<label><input type="checkbox" id="show-neighbors" checked>Selected return and five map neighbors</label>
<label><input type="checkbox" id="show-constraints" checked>Translation constraint axes</label>
<label>Axis scale <input type="number" id="constraint-scale" min="0.1" step="0.1" value="1" style="width:5em"> m</label>
<label><input type="checkbox" id="show-rotation-constraints" checked>Rotation constraint axes</label>
<label>Rotation axis scale <input type="number" id="rotation-scale" min="0.1" step="0.1" value="1" style="width:5em"> m</label>
<label><input type="checkbox" id="show-fast-path" checked>FAST-LIO path</label>
<label><input type="checkbox" id="show-learned-path" checked>Learned odometry path</label>
</div><div class="row"><form id="match-form"><label>Match number
<input id="match-number" type="number" min="1" step="1" value="1"></label>
<button type="submit">Show in map</button></form>
<span id="match-status">Click a matched return or enter its number.</span></div>
<div id="match-plot"></div></main>
<script src="/app.js"></script></body></html>"""


VIEWER_JS = r"""const $=id=>document.getElementById(id);
const request=async(path,options)=>{
  const response=await fetch(path,options);
  const data=await response.json();
  if(!response.ok)throw Error(data.error||response.statusText);
  return data;
};
let scanTimes=[],mapPoints=[],currentScan=null,plotReady=false,selected=null,scanToken=0,matchToken=0;
let selectedResidual=null;
let selectedMatchData=null;
let currentCamera=null;
const mapPlot=$('match-plot'), switches=['map','matched','unmatched','neighbors','constraints',
  'rotation-constraints','fast-path','learned-path']
  .reduce((items,key)=>(items[key]=$('show-'+key),items),{});
function smooth(values,width){
  if(width===1)return values;
  const out=[],queue=[];let sum=0;
  for(const value of values){
    if(value===null||!Number.isFinite(value)){queue.length=0;sum=0;out.push(null);continue}
    queue.push(value);sum+=value;if(queue.length>width)sum-=queue.shift();
    out.push(sum/queue.length);
  }
  return out;
}
const seriesCategories=['pose','matching','constraints','imu','learned','fusion'];
const seriesCache=new Map();
const plotDataCache=new Map();
let activeTab='pose';
let hoverSyncing=false,hoverFrame=0,hoverSource=null,hoverTime=null;
function nearestTimeIndex(times,time){
  let low=0,high=times.length;
  while(low<high){const middle=(low+high)>>1;if(times[middle]<time)low=middle+1;else high=middle}
  if(low===times.length)return times.length-1;
  return low>0&&time-times[low-1]<times[low]-time?low-1:low;
}
function syncSeriesHover(){
  hoverFrame=0;
  const view=seriesCache.get(activeTab);
  if(!view||hoverTime===null)return;
  hoverSyncing=true;
  try{
    for(const graph of view.graphs){
      if(graph===hoverSource||graph.parentElement.hidden||!graph.data?.[0]?.x?.length)continue;
      const pointNumber=nearestTimeIndex(graph.data[0].x,hoverTime);
      Plotly.Fx.hover(graph,[{curveNumber:0,pointNumber}]);
    }
  }finally{hoverSyncing=false}
}
function clearSeriesHover(source){
  if(hoverSyncing||source!==hoverSource)return;
  if(hoverFrame)cancelAnimationFrame(hoverFrame);
  hoverFrame=0;hoverTime=null;hoverSource=null;
  const view=seriesCache.get(activeTab);
  if(!view)return;
  hoverSyncing=true;
  try{for(const graph of view.graphs){if(graph!==source)Plotly.Fx.unhover(graph)}}
  finally{hoverSyncing=false}
}
async function showSeries(category){
  if(hoverFrame)cancelAnimationFrame(hoverFrame);
  hoverFrame=0;hoverTime=null;hoverSource=null;
  activeTab=category;
  $('series-panel').hidden=false;$('match-panel').hidden=true;
  document.querySelectorAll('nav button').forEach(button=>button.classList.toggle('active',
    button.dataset.category===category));
  for(const [key,view] of seriesCache){view.panel.hidden=key!==category}
  if(seriesCache.has(category)){
    requestAnimationFrame(()=>seriesCache.get(category).graphs.forEach(graph=>Plotly.Plots.resize(graph)));
    return;
  }
  $('series-status').textContent='Loading '+category+' plots…';
  try{
    if(!plotDataCache.has(category))plotDataCache.set(category,request('/api/plots/'+category));
    const plots=await plotDataCache.get(category);
    if(seriesCache.has(category))return;
    const panel=document.createElement('div');
    const controls=document.createElement('div');controls.className='controls';
    const charts=document.createElement('div');panel.append(controls,charts);$('plots').append(panel);
    const graphs=[];
    seriesCache.set(category,{panel,graphs});
    panel.hidden=activeTab!==category;
    for(const plot of plots){
    const box=document.createElement('section');
    const heading=document.createElement('h2');heading.textContent=plot.title;box.append(heading);
    const label=document.createElement('label');label.textContent='Smoothing window (samples): ';
    const input=document.createElement('input');input.type='number';input.min=1;input.step=1;input.value=1;
    input.setAttribute('aria-label','Smoothing window for '+plot.title);
    label.append(input);if(plot.key!=='update_valid')box.append(label);
    const graph=document.createElement('div');box.append(graph);charts.append(box);graphs.push(graph);
    const traces=plot.traces.map(t=>({type:t.x.length>2000?'scattergl':'scatter',
      mode:'lines',name:t.name,x:t.x,y:t.y}));
    await Plotly.newPlot(graph,traces,{xaxis:{title:'Seconds from first scan',showspikes:true,
      spikemode:'across',spikesnap:'data'},
      yaxis:{title:plot.unit},height:430,hovermode:'x unified',
      margin:{l:65,r:25,t:30,b:55}},{responsive:true});
    graph.on('plotly_hover',event=>{
      if(hoverSyncing||activeTab!==category||!event.points?.length)return;
      hoverSource=graph;hoverTime=Number(event.points[0].x);
      if(!hoverFrame)hoverFrame=requestAnimationFrame(syncSeriesHover);
    });
    graph.on('plotly_unhover',()=>clearSeriesHover(graph));
    const toggle=document.createElement('label');
    const check=document.createElement('input');check.type='checkbox';check.checked=true;
    check.addEventListener('change',()=>box.hidden=!check.checked);
    toggle.append(check,document.createTextNode(' '+plot.title));controls.append(toggle);
    input.addEventListener('change',()=>{
      const width=Math.max(1,Math.floor(Number(input.value)||1));input.value=width;
      Plotly.restyle(graph,{y:plot.traces.map(t=>smooth(t.y,width))});
    });
    if(activeTab!==category)panel.hidden=true;
    await new Promise(resolve=>requestAnimationFrame(resolve));
    }
    if(!plots.length)charts.textContent='No data available for this tab in this bag.';
    if(activeTab===category)$('series-status').textContent='';
  }catch(error){if(activeTab===category)$('series-status').textContent='Error: '+error.message}
}
async function initSeries(){
  const overview=await request('/api/overview');
  $('summary').textContent=overview.summary;
  $('tab-match').hidden=!overview.has_matches;
  switches['learned-path'].disabled=!overview.has_learned_odometry;
  switches['rotation-constraints'].disabled=!overview.has_pose_information;
  switches.constraints.disabled=!overview.has_translation_constraints;
  if(activeTab!=='match')await showSeries(activeTab);
  // Warm the other tabs without making navigation wait for their first request.
  for(const category of seriesCategories.slice(1)){
    const pending=request('/api/plots/'+category);
    plotDataCache.set(category,pending);
    pending.catch(()=>plotDataCache.delete(category));
  }
}
function trace(points,name,color,size,opacity){
  return {type:'scatter3d',mode:'markers',name,x:points.map(p=>p[0]),
    y:points.map(p=>p[1]),z:points.map(p=>p[2]),marker:{color,size,opacity},
    hovertemplate:name+'<br>x=%{x:.3f} y=%{y:.3f} z=%{z:.3f}<extra></extra>'};
}
function pathTrace(points,name,color,visible){
  const path=points||[];
  return {type:'scatter3d',mode:'lines',name,x:path.map(p=>p[0]),y:path.map(p=>p[1]),
    z:path.map(p=>p[2]),line:{color,width:5},visible:!!points&&visible,
    hovertemplate:name+'<br>x=%{x:.3f} y=%{y:.3f} z=%{z:.3f}<extra></extra>'};
}
function constraintTraces(constraint,start,scale,prefix,colors,visible){
  if(!constraint||!start)return [];
  const total=constraint.values.reduce((sum,value)=>sum+value,0);
  return constraint.values.map((value,i)=>{
    const length=total>0?scale*value/total:0;
    const vector=constraint.vectors[i];
    const minus=start.map((v,j)=>v-length*vector[j]);
    const plus=start.map((v,j)=>v+length*vector[j]);
    return {type:'scatter3d',mode:'lines',name:prefix+' '+['weakest','middle','strongest'][i]+
      ': '+value.toFixed(2),x:[minus[0],plus[0]],y:[minus[1],plus[1]],z:[minus[2],plus[2]],
      line:{color:colors[i],width:5},visible,
      hovertemplate:'%{fullData.name}<extra></extra>'};
  });
}
function summary(){
  if(!currentScan)return;
  $('scan-summary').textContent=currentScan.matched.length+' matched / '+
    (currentScan.matched.length+currentScan.unmatched.length)+' downsampled; '+
    (selected===null?'click a matched return or enter its number':('match '+(selected+1)+
      (selectedResidual===null?' (loading association)':' residual '+Number(selectedResidual).toFixed(4)+' m')));
}
function showSelectedInMap(){
  if(!plotReady)return;
  const match=switches.neighbors.checked?selectedMatchData:null;
  const point=match?match.point:null;
  const neighbors=match?match.neighbors:[];
  const lines=match?neighbors.flatMap(p=>[point,p,null]):[];
  const axes=[0,1,2].map(axis=>[
    neighbors.map(p=>p[axis]),
    lines.map(p=>p===null?null:p[axis]),
    point?[point[axis]]:[]
  ]);
  const first=mapPlot.data.length-3;
  Plotly.update(mapPlot,{x:axes[0],y:axes[1],z:axes[2]},
    currentCamera?{'scene.camera':currentCamera}:{},[first,first+1,first+2]);
}
async function showMatch(index){
  if(!currentScan||!Number.isInteger(index)||index<0||index>=currentScan.matched.length){
    $('match-status').textContent='Choose a match number in this scan.';return;
  }
  selected=index;selectedResidual=null;selectedMatchData=null;summary();
  $('match-number').value=index+1;
  const token=++matchToken,scanIndex=Number($('scan-slider').value);
  $('match-status').textContent='Loading five map neighbors…';
  try{
    const match=await request('/api/scan/'+scanIndex+'/match/'+index);
    if(token!==matchToken||scanIndex!==Number($('scan-slider').value))return;
    selectedMatchData=match;selectedResidual=match.residual;summary();
    $('match-status').textContent='Match '+(index+1)+'; signed residual '+Number(match.residual).toFixed(4)+' m';
    await new Promise(resolve=>requestAnimationFrame(resolve));
    showSelectedInMap();
  }catch(error){$('match-status').textContent='Error: '+error.message}
}
async function renderScan(){
  if(!currentScan)return;
  const scan=currentScan,traces=[];
  const map=trace(mapPoints,'Reference map','#8492a6',1.4,.25);
  map.visible=switches.map.checked;traces.push(map);
  const unmatched=trace(scan.unmatched,'Unmatched returns','#e08b35',2.1,.75);
  unmatched.visible=switches.unmatched.checked;traces.push(unmatched);
  const matched=trace(scan.matched,'Matched returns','#1c9e71',3.3,.95);
  matched.visible=switches.matched.checked;traces.push(matched);
  if(scan.pose){
    traces.push(trace([scan.pose],'Robot position','#2267c5',7,1));
    traces.push({type:'scatter3d',mode:'lines',name:'Robot forward',
      x:[scan.pose[0],scan.pose_tip[0]],y:[scan.pose[1],scan.pose_tip[1]],
      z:[scan.pose[2],scan.pose_tip[2]],line:{color:'#2267c5',width:6},hoverinfo:'skip'});
  }
  traces.push(pathTrace(scan.fast_path,'FAST-LIO estimate','#2456b8',switches['fast-path'].checked));
  traces.push(pathTrace(scan.learned_path,'Learned-velocity odometry','#e26a2c',switches['learned-path'].checked));
  traces.push(...constraintTraces(scan.constraints,scan.pose,
    Math.max(.1,Number($('constraint-scale').value)||1),'Translation constraint',
    ['#8e44ad','#00a6c7','#e74c3c'],switches.constraints.checked));
  traces.push(...constraintTraces(scan.rotation_constraints,scan.pose,
    Math.max(.1,Number($('rotation-scale').value)||1),'Rotation constraint',
    ['#7323a8','#c32b82','#f2a900'],switches['rotation-constraints'].checked));
  traces.push(trace([],'Five map neighbors','#e3265b',3.3,1));
  traces.push({type:'scatter3d',mode:'lines',name:'Associations',
    x:[],y:[],z:[],line:{color:'#e3265b',width:3},hoverinfo:'skip'});
  traces.push(trace([],'Selected return','#ffcc00',3.3,1));
  summary();
  await Plotly.react(mapPlot,traces,{scene:{aspectmode:'data',uirevision:'camera',
    ...(currentCamera?{camera:currentCamera}:{}),xaxis:{title:'map x (m)'},
    yaxis:{title:'map y (m)'},zaxis:{title:'map z (m)'}},
    margin:{l:0,r:0,b:0,t:10},legend:{orientation:'h'},uirevision:'camera'},{responsive:true});
  plotReady=true;
  if(selectedMatchData)showSelectedInMap();
}
async function chooseScan(index){
  if(!Number.isInteger(index)||index<0||index>=scanTimes.length)return;
  const token=++scanToken;++matchToken;
  selected=null;selectedResidual=null;selectedMatchData=null;
  $('match-status').textContent='Click a matched return or enter its number.';
  $('scan-slider').value=index;$('scan-select').value=index;
  $('scan-prev').disabled=index===0;
  $('scan-next').disabled=index===scanTimes.length-1;
  $('scan-summary').textContent='Loading scan…';
  try{
    const scan=await request('/api/scan/'+index);
    if(token!==scanToken)return;
    currentScan=scan;await renderScan();
  }catch(error){$('scan-summary').textContent='Error: '+error.message}
}
async function initMatches(){
  const index=await request('/api/scans');scanTimes=index.times;
  $('scan-slider').max=Math.max(0,scanTimes.length-1);
  $('scan-prev').disabled=true;$('scan-next').disabled=scanTimes.length<2;
  for(let i=0;i<scanTimes.length;i++){
    const option=document.createElement('option');option.value=i;
    option.textContent=scanTimes[i].toFixed(3)+' s';$('scan-select').append(option);
  }
  const map=await request('/api/map');mapPoints=map.points;
  $('map-status').textContent=map.count?map.count+' displayed map points':'Choose a reference PCD to show the map.';
  if(scanTimes.length)await chooseScan(0);
  mapPlot.on('plotly_click',event=>{
    const hit=event.points[0];
    if(hit.curveNumber===2){
      const index=Number(hit.pointNumber ?? hit.pointIndex);
      showMatch(index);
    }
  });
  mapPlot.on('plotly_relayout',update=>{
    if(update['scene.camera'])currentCamera=JSON.parse(JSON.stringify(update['scene.camera']));
  });
}
document.querySelectorAll('.series-tab').forEach(button=>button.addEventListener('click',
  ()=>showSeries(button.dataset.category)));
let matchesInitialized=false;
$('tab-match').addEventListener('click',async()=>{
  $('series-panel').hidden=true;$('match-panel').hidden=false;
  activeTab='match';
  document.querySelectorAll('nav button').forEach(button=>button.classList.toggle('active',button.id==='tab-match'));
  if(!matchesInitialized){matchesInitialized=true;try{await initMatches()}catch(error){$('scan-summary').textContent='Error: '+error.message}}
  else if(plotReady)Plotly.Plots.resize(mapPlot);
});
$('scan-slider').addEventListener('change',event=>chooseScan(Number(event.target.value)));
$('scan-select').addEventListener('change',event=>chooseScan(Number(event.target.value)));
$('scan-prev').addEventListener('click',()=>chooseScan(Number($('scan-slider').value)-1));
$('scan-next').addEventListener('click',()=>chooseScan(Number($('scan-slider').value)+1));
function setTraceVisibility(key){
  if(!plotReady)return;
  const prefixes={map:'Reference map',matched:'Matched returns',unmatched:'Unmatched returns',
    constraints:'Translation constraint','rotation-constraints':'Rotation constraint',
    'fast-path':'FAST-LIO estimate','learned-path':'Learned-velocity odometry'};
  const indices=mapPlot.data.flatMap((trace,index)=>trace.name.startsWith(prefixes[key])?[index]:[]);
  if(indices.length)Plotly.restyle(mapPlot,{visible:switches[key].checked},indices);
}
for(const [key,input] of Object.entries(switches)){
  input.addEventListener('change',key==='neighbors'?showSelectedInMap:()=>setTraceVisibility(key));
}
$('constraint-scale').addEventListener('change',renderScan);
$('rotation-scale').addEventListener('change',renderScan);
$('match-form').addEventListener('submit',event=>{
  event.preventDefault();showMatch(Number($('match-number').value)-1);
});
$('pcd-file').addEventListener('change',async event=>{
  const file=event.target.files?.[0];if(!file)return;
  $('map-status').textContent='Loading '+file.name+'…';
  try{
    const map=await request('/api/map',{method:'POST',headers:{'Content-Type':'application/octet-stream'},
      body:file});mapPoints=map.points;
    $('map-status').textContent=file.name+': '+map.count+' displayed points';
    await renderScan();
  }catch(error){$('map-status').textContent='Error: '+error.message}
});
$('pcd-path-form').addEventListener('submit',async event=>{
  event.preventDefault();
  const path=$('pcd-path').value.trim();if(!path)return;
  $('map-status').textContent='Loading '+path+'…';
  try{
    const map=await request('/api/map-path',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify({path})});
    mapPoints=map.points;
    $('map-status').textContent=path+': '+map.count+' displayed points';
    await renderScan();
  }catch(error){$('map-status').textContent='Error: '+error.message}
});
initSeries().catch(error=>{$('summary').textContent='Error: '+error.message});
"""


def serve_viewer(data, port, open_browser):
    from plotly.offline import get_plotlyjs

    plotly_js = get_plotlyjs().encode("utf-8")
    overview = json.dumps({key: value for key, value in data.overview.items() if key != "plots"},
                          separators=(",", ":"), allow_nan=False).encode()
    plot_payloads = {category: json.dumps([plot for plot in data.overview["plots"]
                                           if plot["category"] == category],
                                          separators=(",", ":"), allow_nan=False).encode()
                     for category in ("pose", "matching", "constraints", "imu", "learned", "fusion")}

    class Handler(BaseHTTPRequestHandler):
        def respond(self, body, content_type="application/json", status=200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def json_response(self, value, status=200):
            self.respond(json.dumps(value, separators=(",", ":"), allow_nan=False).encode(), status=status)

        def do_GET(self):
            path = urlparse(self.path).path
            try:
                if path == "/":
                    self.respond(VIEWER_HTML.encode(), "text/html; charset=utf-8")
                elif path == "/app.js":
                    self.respond(VIEWER_JS.encode(), "text/javascript; charset=utf-8")
                elif path == "/plotly.js":
                    self.respond(plotly_js, "text/javascript; charset=utf-8")
                elif path == "/api/overview":
                    self.respond(overview)
                elif path.startswith("/api/plots/") and path.removeprefix("/api/plots/") in plot_payloads:
                    self.respond(plot_payloads[path.removeprefix("/api/plots/")])
                elif path == "/api/scans":
                    self.json_response(data.scan_index())
                elif path == "/api/map":
                    self.json_response(data.map_payload())
                elif path.startswith("/api/scan/"):
                    parts = path.split("/")
                    if len(parts) not in (4, 6) or (len(parts) == 6 and parts[4] != "match"):
                        raise ValueError("Invalid scan request")
                    index = int(parts[3])
                    if not 0 <= index < len(data.scans):
                        raise ValueError("Scan index out of range")
                    if len(parts) == 6:
                        self.json_response(data.match_payload(index, int(parts[5])))
                    else:
                        self.json_response(data.scan_payload(index))
                else:
                    self.json_response({"error": "Not found"}, 404)
            except (ValueError, IndexError) as error:
                self.json_response({"error": str(error)}, 400)

        def do_POST(self):
            path = urlparse(self.path).path
            if path not in ("/api/map", "/api/map-path"):
                self.json_response({"error": "Not found"}, 404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if path == "/api/map-path":
                    if not 0 < length <= 4096:
                        raise ValueError("PCD path request is empty or too long")
                    requested = json.loads(self.rfile.read(length))
                    map_path = Path(requested["path"]).expanduser()
                    with map_path.open("rb") as stream:
                        count = data.set_map(stream)
                else:
                    if not 0 < length <= 256*1024*1024:
                        raise ValueError("PCD must be between 1 byte and 256 MB")
                    count = data.set_map(io.BytesIO(self.rfile.read(length)))
                self.json_response(data.map_payload())
                print(f"Loaded reference PCD: {count} displayed points")
            except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as error:
                self.json_response({"error": str(error)}, 400)

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"FAST-LIO viewer: {url}")
    print("Press Ctrl+C to stop the server.")
    if open_browser:
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def select_paths_gui(args):
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    try:
        root = tk.Tk()
    except tk.TclError as error:
        raise ValueError(f"Cannot open file picker: {error}. Pass BAG_DIR on the command line") from error
    root.title("FAST-LIO analyzer")
    root.geometry("720x225")
    frame = ttk.Frame(root, padding=18)
    frame.pack(fill="both", expand=True)
    frame.columnconfigure(1, weight=1)
    bag_var = tk.StringVar(value=str(args.bag or ""))
    map_var = tk.StringVar(value=str(args.map_pcd or ""))
    chosen = {}

    def choose_bag():
        path = filedialog.askdirectory(title="Choose ROS 2 bag folder", initialdir=str(Path.cwd()))
        if path:
            bag_var.set(path)

    def choose_map():
        path = filedialog.askopenfilename(title="Choose reference PCD", initialdir=str(Path.cwd()),
                                          filetypes=[("PCD point clouds", "*.pcd"), ("All files", "*")])
        if path:
            map_var.set(path)

    for row, (label, variable, command) in enumerate((
        ("ROS 2 bag folder", bag_var, choose_bag),
        ("Reference PCD (optional)", map_var, choose_map),
    )):
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=8)
        ttk.Entry(frame, textvariable=variable).grid(row=row, column=1, sticky="ew", pady=8)
        ttk.Button(frame, text="Browse…", command=command).grid(row=row, column=2, padx=(10, 0), pady=8)
    ttk.Label(frame, text="The browser can also load or replace the PCD after the viewer opens.").grid(
        row=2, column=0, columnspan=3, sticky="w", pady=10)

    def launch():
        bag = Path(bag_var.get().strip()).expanduser()
        map_pcd = Path(map_var.get().strip()).expanduser() if map_var.get().strip() else None
        if not (bag / "metadata.yaml").is_file():
            messagebox.showerror("FAST-LIO analyzer", "Choose a bag folder containing metadata.yaml.")
            return
        if map_pcd is not None and not map_pcd.is_file():
            messagebox.showerror("FAST-LIO analyzer", "Choose an existing PCD file.")
            return
        chosen.update(bag=bag, map_pcd=map_pcd)
        root.destroy()

    ttk.Button(frame, text="Open viewer", command=launch).grid(row=3, column=2, sticky="e", pady=10)
    root.mainloop()
    return chosen or None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bag", nargs="?", type=Path, help="ROS 2 bag directory; omit to choose in a file picker")
    parser.add_argument("--map-pcd", type=Path, help="Optional reference PCD to load at startup")
    parser.add_argument("--map-point-limit", type=int, default=8000,
                        help="Maximum displayed reference-map points (default: 8000)")
    parser.add_argument("--port", type=int, default=0, help="Local server port; 0 chooses a free port")
    parser.add_argument("--no-open", action="store_true", help="Print URL without opening a browser")
    parser.add_argument("--diagnostics-topic", default=DIAGNOSTICS_TOPIC)
    parser.add_argument("--imu-topic", default=BODY_IMU_TOPIC, help="Must be the Go2 body IMU")
    parser.add_argument("--odometry-topic", default=ODOMETRY_TOPIC)
    parser.add_argument("--match-debug-topic", default=MATCH_DEBUG_TOPIC)
    args = parser.parse_args()
    try:
        if args.bag is None:
            selected = select_paths_gui(args)
            if selected is None:
                return
            args.bag, args.map_pcd = selected["bag"], selected["map_pcd"]
        diagnostics, imu, odometry, matches, tf_messages, pose_information, learned_odometry, learned_velocity, velocity_fusion = read_bag(
            args.bag, args.diagnostics_topic, args.imu_topic, args.odometry_topic, args.match_debug_topic)
        data = ViewerData(diagnostics, imu, odometry, matches, tf_messages, args.map_pcd,
                          args.map_point_limit, pose_information, learned_odometry, learned_velocity, velocity_fusion)
        serve_viewer(data, args.port, not args.no_open)
    except (ValueError, ImportError, KeyError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
