#!/usr/bin/env python3
"""Replay recorded FAST-LIO inputs into one new estimate bag.

From source/unitree_mujoco/ros2_ws, after sourcing ROS and install/setup.bash:
    python3 src/go2_dds_ros2_bridge/scripts/replay_fast_lio_bag.py INPUT_BAG \
      --mode imu_lidar_learned_velocity --output learned_result

Use --inspect to check a bag without launching ROS processes. Paths are relative
to the terminal's working directory. The original bag is never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

import yaml


DEFAULT_CLOUD = "/utlidar/time_corrected/cloud"
DEFAULT_IMU = "/imu/data_fastlio"
DEFAULT_VELOCITY = "/estimated_velocity"
RESULT_TOPICS = [
    "/Odometry", "/path", "/cloud_registered", "/Laser_map", "/tf",
    "/fast_lio/estimator_diagnostics", "/fast_lio/pose_information",
    "/fast_lio/match_debug", "/fast_lio/velocity_fusion_debug",
]


def inspect_bag(path: Path) -> dict:
    metadata = path / "metadata.yaml"
    if not metadata.is_file():
        raise ValueError(f"No metadata.yaml in {path}; paths use the terminal's working directory")
    info = yaml.safe_load(metadata.read_text())["rosbag2_bagfile_information"]
    return {item["topic_metadata"]["name"]: item for item in info["topics_with_message_count"]}


def fusion_status_counts(path: Path) -> dict[str, int]:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from fast_lio.msg import VelocityFusionDebug

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(path), storage_id="sqlite3"),
                rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"))
    labels = {0: "accepted", 1: "late", 2: "invalid_or_duplicate", 3: "numerical_failure"}
    counts = {name: 0 for name in labels.values()}
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic == "/fast_lio/velocity_fusion_debug":
            status = deserialize_message(data, VelocityFusionDebug).status
            counts[labels.get(status, "unknown")] = counts.get(labels.get(status, "unknown"), 0) + 1
    return counts


def commit_at(path: Path) -> str | None:
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stop_process(process: subprocess.Popen | None, timeout: float = 15) -> None:
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def wait_graph(node, predicate, processes, label: str, timeout: float = 20) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for process in processes:
            if process.poll() is not None:
                raise RuntimeError(f"{label}: a ROS process exited early; see replay_process.log")
        if predicate():
            return
        time.sleep(0.1)
    raise RuntimeError(f"Timed out waiting for {label}; see replay_process.log")


def run(args) -> None:
    script = Path(__file__).resolve()
    workspace = script.parents[3]
    bridge_src = script.parents[1]
    sys.path.insert(0, str(bridge_src))
    from go2_dds_ros2_bridge.tf_utils import (base_link_pose_in_imu_frame,
                                                flatten_rotation_matrix, lidar_pose_in_imu_frame)
    import rclpy
    from nav_msgs.msg import Odometry

    output = args.output.expanduser().resolve()
    bag_info = yaml.safe_load((args.bag / "metadata.yaml").read_text())["rosbag2_bagfile_information"]
    duration = bag_info["duration"]["nanoseconds"] / 1e9
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    config_path = args.config.expanduser().resolve() if args.config else bridge_src / "config/fast_lio.yaml"
    params = yaml.safe_load(config_path.read_text())
    values = params["/**"]["ros__parameters"]
    lidar_t, lidar_r = lidar_pose_in_imu_frame()
    base_t, base_r = base_link_pose_in_imu_frame()
    values["common"]["lid_topic"] = args.cloud_topic
    values["common"]["imu_topic"] = args.imu_topic
    values["learned_velocity"].update(
        mode=args.mode, topic=args.velocity_topic, stddev=args.sigma,
        base_in_imu_translation=[float(v) for v in base_t],
        base_in_imu_rotation=flatten_rotation_matrix(base_r))
    values["mapping"]["extrinsic_est_en"] = False
    values["mapping"]["extrinsic_T"] = [float(v) for v in lidar_t]
    values["mapping"]["extrinsic_R"] = flatten_rotation_matrix(lidar_r)
    values.setdefault("publish", {})["match_debug_en"] = True
    values.setdefault("pcd_save", {})["pcd_save_en"] = False
    values["use_sim_time"] = True
    output.parent.mkdir(parents=True, exist_ok=True)
    optional_inputs = [topic for topic in (args.raw_imu_topic, "/odom", "/tf_static") if topic in args.topics]
    play_topics = [args.cloud_topic, args.imu_topic]
    if args.velocity_topic in args.topics:
        play_topics.append(args.velocity_topic)
    play_topics += optional_inputs
    record_topics = RESULT_TOPICS + [args.imu_topic] + (
        [args.velocity_topic] if args.velocity_topic in args.topics else []) + optional_inputs
    manifest = {
        "input_bag": str(args.bag), "output_bag": str(output), "mode": args.mode,
        "input_metadata_sha256": sha256(args.bag / "metadata.yaml"),
        "replay_rate": args.rate, "played_topics": play_topics, "recorded_topics": record_topics,
        "fast_lio_commit": commit_at(workspace / "src/FAST_LIO"),
        "bridge_commit": commit_at(bridge_src), "input_topic_counts":
        {topic: args.topics[topic]["message_count"] for topic in play_topics},
        "replay_script_sha256": sha256(script),
        "analyzer_script_sha256": sha256(script.with_name("analyze_fast_lio_bag.py")),
    }
    environment = os.environ.copy()
    environment["ROS_DOMAIN_ID"] = str(args.domain_id)
    os.environ["ROS_DOMAIN_ID"] = str(args.domain_id)
    children = []
    rclpy.init(args=None)
    node = rclpy.create_node("fast_lio_replay_monitor")
    last_estimate = [time.monotonic()]
    node.create_subscription(Odometry, "/Odometry",
                             lambda _: last_estimate.__setitem__(0, time.monotonic()), 100)
    try:
        with tempfile.TemporaryDirectory(prefix="fast_lio_replay_") as temporary:
            parameters = Path(temporary) / "parameters.yaml"
            parameters.write_text(yaml.safe_dump(params, sort_keys=False))
            with (output.parent / f"{output.name}.replay_process.log").open("w") as log:
                def start(command):
                    print("Starting:", " ".join(map(str, command)), flush=True)
                    process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                    children.append(process)
                    return process

                fast = start(["ros2", "run", "fast_lio", "fastlio_mapping", "--ros-args",
                              "--params-file", str(parameters)])
                recorder = start(["ros2", "bag", "record", "-o", str(output), "--use-sim-time",
                                  *record_topics])
                wait_graph(node, lambda: all(node.count_subscribers(topic) > 0
                                             for topic in play_topics[:2] +
                                             ([args.velocity_topic] if args.mode == "imu_lidar_learned_velocity" else [])),
                           (fast, recorder), "FAST-LIO input subscriptions")
                # rosbag2's --use-sim-time recorder waits for the first /clock
                # before subscribing, so its subscriptions cannot be a pre-play gate.
                player = start(["ros2", "bag", "play", str(args.bag), "--clock", "100",
                                "--rate", str(args.rate), "--delay", "2",
                                "--disable-keyboard-controls",
                                "--topics", *play_topics])
                playback_started = time.monotonic() + 2  # ros2 bag play --delay 2
                last_report = -1
                while player.poll() is None:
                    rclpy.spin_once(node, timeout_sec=0.1)
                    percent = min(99, int(100 * max(0, time.monotonic()-playback_started)
                                          * args.rate / max(duration, 0.001)))
                    if percent >= last_report + 5:
                        print(f"Playback: {percent}%", flush=True)
                        last_report = percent
                    if fast.poll() is not None or recorder.poll() is not None:
                        raise RuntimeError("FAST-LIO or recorder exited during playback; see replay_process.log")
                if player.returncode != 0:
                    raise RuntimeError("Bag playback failed; see replay_process.log")
                print("Playback: 100% — processing final scans", flush=True)
                # Wait for output to settle; heavy scans can finish after playback.
                last_estimate[0] = time.monotonic()
                drain_deadline = time.monotonic() + 30
                while time.monotonic() < drain_deadline and time.monotonic()-last_estimate[0] < 3:
                    rclpy.spin_once(node, timeout_sec=0.1)
                if time.monotonic()-last_estimate[0] < 3:
                    print("Warning: output was still active after the 30 s drain limit", flush=True)
                stop_process(recorder)
                print("Result bag finalized", flush=True)
                stop_process(fast)
        if not (output / "metadata.yaml").is_file():
            raise RuntimeError("Recorder did not finalize the result bag")
        result_topics = inspect_bag(output)
        manifest["result_topic_counts"] = {key: value["message_count"] for key, value in result_topics.items()}
        if args.mode == "imu_lidar_learned_velocity":
            manifest["velocity_fusion_status_counts"] = fusion_status_counts(output)
        if manifest["result_topic_counts"].get("/Odometry", 0) == 0:
            raise RuntimeError("Replay produced no /Odometry; inspect replay_process.log and the partial result bag")
        (output / "replay_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (output / "effective_parameters.yaml").write_text(yaml.safe_dump(params, sort_keys=False))
        print(f"Result: {output} ({manifest['result_topic_counts']['/Odometry']} estimates)")
        if "velocity_fusion_status_counts" in manifest:
            print("Velocity fusion:", manifest["velocity_fusion_status_counts"])
    finally:
        for process in reversed(children):
            stop_process(process)
        node.destroy_node()
        rclpy.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bag", type=Path, help="Input ROS bag directory")
    parser.add_argument("--output", type=Path, help="New result bag directory")
    parser.add_argument("--mode", choices=("imu_lidar", "imu_lidar_learned_velocity"), default="imu_lidar")
    parser.add_argument("--inspect", action="store_true", help="Show topic counts and replay readiness; do not replay")
    parser.add_argument("--rate", type=float, default=1.0, help="Playback rate; default 1.0")
    parser.add_argument("--domain-id", type=int, default=170, help="Isolated ROS domain; default 170")
    parser.add_argument("--config", type=Path, help="FAST-LIO YAML; default bridge config/fast_lio.yaml")
    parser.add_argument("--cloud-topic", default=DEFAULT_CLOUD)
    parser.add_argument("--imu-topic", default=DEFAULT_IMU)
    parser.add_argument("--raw-imu-topic", default=f"{DEFAULT_IMU}_raw")
    parser.add_argument("--velocity-topic", default=DEFAULT_VELOCITY)
    parser.add_argument("--sigma", type=float, nargs=3, default=[0.5, 0.5, 0.5], metavar=("X", "Y", "Z"))
    args = parser.parse_args()
    try:
        if args.rate <= 0 or any(not 0 < value < float("inf") for value in args.sigma):
            raise ValueError("Replay rate and velocity standard deviations must be finite and positive")
        args.bag = args.bag.expanduser().resolve()
        args.topics = inspect_bag(args.bag)
        required = {args.cloud_topic: "sensor_msgs/msg/PointCloud2", args.imu_topic: "sensor_msgs/msg/Imu"}
        if args.mode == "imu_lidar_learned_velocity":
            required[args.velocity_topic] = "geometry_msgs/msg/TwistStamped"
        for topic, expected_type in required.items():
            item = args.topics.get(topic)
            count = item["message_count"] if item else 0
            actual_type = item["topic_metadata"]["type"] if item else None
            print(f"{topic}: {count} messages" + (f" ({actual_type})" if item else " (missing)"))
        missing = [name for name, expected in required.items()
                   if name not in args.topics or args.topics[name]["message_count"] == 0 or
                   args.topics[name]["topic_metadata"]["type"] != expected]
        if missing:
            raise ValueError("Cannot replay: required input topic missing, empty, or wrong type: " + ", ".join(missing))
        if args.inspect:
            print("Bag is ready for replay in", args.mode, "mode")
            return
        if args.output is None:
            raise ValueError("--output is required to run a replay")
        run(args)
    except (ValueError, KeyError, OSError, RuntimeError, ImportError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
