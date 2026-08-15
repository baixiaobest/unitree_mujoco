#!/usr/bin/env python3
"""Quantify temporal-lidar instability and its relationship to policy commands.

The script reads the policy-facing ``TemporalLidarObservation`` stream directly.
It distinguishes updates that keep the same completed scan (only current-pose
reprojection/front-arc selection can change) from updates that install a new
completed scan.  If a goal is present, samples before the first goal are
reported as an idle baseline and later samples as the moving segment.

Run from a ROS environment that can resolve ``go2_dds_ros2_bridge_msgs``::

    python3 tools/analyze_temporal_lidar_bag.py bags/live_lidar_straight_02 \
      --output-dir bags/live_lidar_straight_02/analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


OBSERVATION_TOPIC = "/temporal_lidar/observation"
COMMAND_TOPIC = "/cmd_vel"
GOAL_TOPIC = "/goal_pose"
DEFAULT_HISTORY_FRAMES = 4
LARGE_CHANGE_THRESHOLD = 0.1  # Normalized range = 2 m at a 20 m lidar range.


def _float(value: np.generic | float) -> float:
    return float(value)


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"mean": math.nan, "p50": math.nan, "p95": math.nan, "max": math.nan}
    return {
        "mean": _float(np.mean(values)),
        "p50": _float(np.quantile(values, 0.50)),
        "p95": _float(np.quantile(values, 0.95)),
        "max": _float(np.max(values)),
    }


def _read_bag(bag_path: Path) -> dict[str, list[tuple[int, Any]]]:
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    message_types = {item.name: get_message(item.type) for item in reader.get_all_topics_and_types()}
    wanted_topics = {OBSERVATION_TOPIC, COMMAND_TOPIC, GOAL_TOPIC}
    messages: dict[str, list[tuple[int, Any]]] = defaultdict(list)
    while reader.has_next():
        topic, serialized, timestamp_ns = reader.read_next()
        if topic in wanted_topics:
            messages[topic].append((timestamp_ns, deserialize_message(serialized, message_types[topic])))
    return messages


def _metric_for_transitions(
    transition_mask: np.ndarray,
    distance_change: np.ndarray,
    jointly_valid: np.ndarray,
    validity_flips: np.ndarray,
) -> dict[str, Any]:
    """Summarize selected observation-to-observation transitions."""
    selected_change = distance_change[transition_mask]
    selected_jointly_valid = jointly_valid[transition_mask]
    selected_flips = validity_flips[transition_mask]
    values = selected_change[selected_jointly_valid]
    result: dict[str, Any] = {
        "transition_count": int(np.count_nonzero(transition_mask)),
        "jointly_valid_fraction": _float(np.mean(selected_jointly_valid)) if selected_jointly_valid.size else math.nan,
        "validity_flip_fraction": _float(np.mean(selected_flips)) if selected_flips.size else math.nan,
        "large_distance_change_fraction": _float(np.mean(values > LARGE_CHANGE_THRESHOLD)) if values.size else math.nan,
        "distance_change": _quantiles(values),
        "per_frame": [],
    }
    for frame in range(distance_change.shape[1]):
        frame_mask = selected_jointly_valid[:, frame, :]
        frame_values = selected_change[:, frame, :][frame_mask]
        result["per_frame"].append(
            {
                "frame": frame,
                "jointly_valid_fraction": _float(np.mean(frame_mask)) if frame_mask.size else math.nan,
                "large_distance_change_fraction": _float(np.mean(frame_values > LARGE_CHANGE_THRESHOLD))
                if frame_values.size
                else math.nan,
                "distance_change": _quantiles(frame_values),
            }
        )
    return result


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return math.nan
    return _float(np.corrcoef(x, y)[0, 1])


def _command_correlation(
    observation_times_ns: np.ndarray,
    distance_change: np.ndarray,
    jointly_valid: np.ndarray,
    command_times_ns: np.ndarray,
    commands: np.ndarray,
) -> dict[str, Any]:
    if command_times_ns.size < 2:
        return {"matched_command_deltas": 0}
    command_delta = np.linalg.norm(np.diff(commands, axis=0), axis=1)
    command_delta_times = command_times_ns[1:]
    observation_indices = np.searchsorted(observation_times_ns, command_delta_times, side="right") - 1
    valid = (observation_indices >= 1) & (
        command_delta_times - observation_times_ns[np.maximum(observation_indices, 0)] <= 200_000_000
    )
    indices = observation_indices[valid]
    if indices.size == 0:
        return {"matched_command_deltas": 0}

    selected_change = distance_change[indices - 1]
    selected_valid = jointly_valid[indices - 1]
    mean_change = np.array(
        [changes[mask].mean() if np.any(mask) else math.nan for changes, mask in zip(selected_change, selected_valid)],
        dtype=np.float64,
    )
    large_fraction = np.array(
        [np.mean(changes[mask] > LARGE_CHANGE_THRESHOLD) if np.any(mask) else math.nan
         for changes, mask in zip(selected_change, selected_valid)],
        dtype=np.float64,
    )
    finite = np.isfinite(mean_change) & np.isfinite(large_fraction)
    return {
        "matched_command_deltas": int(np.count_nonzero(finite)),
        "command_delta_l2": _quantiles(command_delta[valid][finite]),
        "correlation_command_delta_vs_mean_lidar_distance_change": _pearson(
            command_delta[valid][finite], mean_change[finite]
        ),
        "correlation_command_delta_vs_large_lidar_change_fraction": _pearson(
            command_delta[valid][finite], large_fraction[finite]
        ),
        "per_axis_sign_flip_fraction": [
            _float(value) for value in np.mean(commands[1:] * commands[:-1] < 0.0, axis=0)
        ],
    }


def _write_timeline(
    output_dir: Path,
    observation_times_ns: np.ndarray,
    scan_stamps_ns: np.ndarray,
    scan_ages: np.ndarray,
    new_scan: np.ndarray,
    distance_change: np.ndarray,
    jointly_valid: np.ndarray,
    validity_flips: np.ndarray,
) -> None:
    """Write one row per lidar transition for quick plotting in a spreadsheet."""
    with (output_dir / "temporal_lidar_timeline.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "time_s", "scan_stamp_s", "normalized_scan_age", "new_completed_scan",
                "mean_distance_change", "p95_distance_change", "large_change_fraction",
                "validity_flip_fraction",
            ),
        )
        writer.writeheader()
        start_ns = observation_times_ns[0]
        for index in range(1, observation_times_ns.size):
            changes = distance_change[index - 1]
            valid = jointly_valid[index - 1]
            values = changes[valid]
            writer.writerow(
                {
                    "time_s": (observation_times_ns[index] - start_ns) / 1e9,
                    "scan_stamp_s": scan_stamps_ns[index] / 1e9,
                    "normalized_scan_age": scan_ages[index],
                    "new_completed_scan": int(new_scan[index - 1]),
                    "mean_distance_change": float(np.mean(values)) if values.size else "",
                    "p95_distance_change": float(np.quantile(values, 0.95)) if values.size else "",
                    "large_change_fraction": float(np.mean(values > LARGE_CHANGE_THRESHOLD)) if values.size else "",
                    "validity_flip_fraction": float(np.mean(validity_flips[index - 1])),
                }
            )


def analyze(bag_path: Path, *, history_frames: int, motion_start_sec: float | None) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    messages = _read_bag(bag_path)
    observations = messages[OBSERVATION_TOPIC]
    if len(observations) < 2:
        raise RuntimeError("Bag needs at least two TemporalLidarObservation messages.")

    observation_times_ns = np.asarray([timestamp for timestamp, _ in observations], dtype=np.int64)
    observation_messages = [message for _, message in observations]
    flat_size = len(observation_messages[0].distances)
    if flat_size == 0 or flat_size % history_frames:
        raise RuntimeError(f"Unexpected temporal-lidar distance length {flat_size} for H={history_frames}.")
    bins = flat_size // history_frames
    if any(len(message.distances) != flat_size or len(message.validity) != flat_size for message in observation_messages):
        raise RuntimeError("Temporal lidar message array sizes are inconsistent within this bag.")

    distances = np.asarray([message.distances for message in observation_messages], dtype=np.float32).reshape(-1, history_frames, bins)
    validity = np.asarray([message.validity for message in observation_messages], dtype=np.uint8).reshape(-1, history_frames, bins)
    scan_stamps_ns = np.asarray(
        [message.scan_stamp.sec * 1_000_000_000 + message.scan_stamp.nanosec for message in observation_messages],
        dtype=np.int64,
    )
    header_stamps_ns = np.asarray(
        [message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec for message in observation_messages],
        dtype=np.int64,
    )
    scan_ages = np.asarray([message.normalized_scan_age for message in observation_messages], dtype=np.float32)

    distance_change = np.abs(np.diff(distances, axis=0))
    jointly_valid = (validity[1:] > 0) & (validity[:-1] > 0)
    validity_flips = validity[1:] != validity[:-1]
    new_scan = scan_stamps_ns[1:] != scan_stamps_ns[:-1]

    if motion_start_sec is not None:
        motion_start_ns = observation_times_ns[0] + int(motion_start_sec * 1e9)
        motion_start_source = "--motion-start-sec"
    elif messages[GOAL_TOPIC]:
        motion_start_ns = messages[GOAL_TOPIC][0][0]
        motion_start_source = "first /goal_pose"
    elif messages[COMMAND_TOPIC]:
        motion_start_ns = messages[COMMAND_TOPIC][0][0]
        motion_start_source = "first /cmd_vel"
    else:
        motion_start_ns = None
        motion_start_source = "unavailable"

    transition_times_ns = observation_times_ns[1:]
    segments: dict[str, np.ndarray] = {"all": np.ones_like(new_scan, dtype=bool)}
    if motion_start_ns is not None:
        segments["idle_before_goal"] = transition_times_ns < motion_start_ns
        segments["after_goal"] = transition_times_ns >= motion_start_ns

    segment_results: dict[str, Any] = {}
    for name, segment in segments.items():
        segment_results[name] = {
            "same_completed_scan": _metric_for_transitions(
                segment & ~new_scan, distance_change, jointly_valid, validity_flips
            ),
            "new_completed_scan": _metric_for_transitions(
                segment & new_scan, distance_change, jointly_valid, validity_flips
            ),
        }

    command_times_ns = np.asarray([timestamp for timestamp, _ in messages[COMMAND_TOPIC]], dtype=np.int64)
    commands = np.asarray(
        [[message.twist.linear.x, message.twist.linear.y, message.twist.angular.z] for _, message in messages[COMMAND_TOPIC]],
        dtype=np.float64,
    )
    command_results = _command_correlation(
        observation_times_ns, distance_change, jointly_valid, command_times_ns, commands
    )

    unique_scan_stamps = np.unique(scan_stamps_ns)
    summary: dict[str, Any] = {
        "bag": str(bag_path),
        "observation_count": int(observation_times_ns.size),
        "observation_duration_s": _float((observation_times_ns[-1] - observation_times_ns[0]) / 1e9),
        "observation_rate_hz": _float((observation_times_ns.size - 1) * 1e9 / (observation_times_ns[-1] - observation_times_ns[0])),
        "history_frames": history_frames,
        "fov_bins": bins,
        "completed_scan_count": int(unique_scan_stamps.size),
        "completed_scan_rate_hz": _float(
            (unique_scan_stamps.size - 1) * 1e9 / (unique_scan_stamps[-1] - unique_scan_stamps[0])
        ) if unique_scan_stamps.size > 1 else math.nan,
        "normalized_scan_age": _quantiles(scan_ages),
        "scan_transport_latency_s": _quantiles((header_stamps_ns - scan_stamps_ns) / 1e9),
        "valid_fraction_per_history_frame": [_float(value) for value in np.mean(validity > 0, axis=(0, 2))],
        "motion_start_source": motion_start_source,
        "motion_start_from_first_observation_s": _float((motion_start_ns - observation_times_ns[0]) / 1e9)
        if motion_start_ns is not None else math.nan,
        "segments": segment_results,
        "command_correlation": command_results,
    }
    arrays = {
        "observation_times_ns": observation_times_ns,
        "scan_stamps_ns": scan_stamps_ns,
        "scan_ages": scan_ages,
        "new_scan": new_scan,
        "distance_change": distance_change,
        "jointly_valid": jointly_valid,
        "validity_flips": validity_flips,
    }
    return summary, arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path, help="Directory containing metadata.yaml and the SQLite rosbag.")
    parser.add_argument("--history-frames", type=int, default=DEFAULT_HISTORY_FRAMES)
    parser.add_argument("--motion-start-sec", type=float, default=None,
                        help="Override automatic goal-based segmentation; seconds after first lidar observation.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Write summary.json and a per-transition CSV to this directory.")
    args = parser.parse_args()
    if args.history_frames <= 0:
        parser.error("--history-frames must be positive")

    summary, arrays = analyze(args.bag, history_frames=args.history_frames, motion_start_sec=args.motion_start_sec)
    print(json.dumps(summary, indent=2, allow_nan=True))
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / "summary.json").open("w") as stream:
            json.dump(summary, stream, indent=2, allow_nan=True)
            stream.write("\n")
        _write_timeline(args.output_dir, **arrays)
        print(f"Wrote {args.output_dir / 'summary.json'}")
        print(f"Wrote {args.output_dir / 'temporal_lidar_timeline.csv'}")


if __name__ == "__main__":
    main()
