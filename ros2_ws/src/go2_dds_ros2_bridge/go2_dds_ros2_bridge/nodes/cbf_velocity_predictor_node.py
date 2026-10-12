#!/usr/bin/env python3
"""Prepare matched obstacle velocities outside the CBF control timer."""

from threading import Condition, Thread
from time import monotonic

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from go2_dds_ros2_bridge.cbf_dynamic import CONTRACT, ScanHistory, history_arrays, prepare_points, project_history
from go2_dds_ros2_bridge_msgs.msg import CbfControlSnapshot, CbfDynamicSnapshot, CbfWorldScan


def stamp_ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def load_predictor(path):
    """Read the deployment contract and verify a model at its declared shape."""
    import torch

    metadata = {"projection_contract.txt": b"", "num_frames.txt": b""}
    model = torch.jit.load(path, map_location="cpu", _extra_files=metadata).eval()
    contract = metadata["projection_contract.txt"]
    frames_text = metadata["num_frames.txt"]
    if (contract == b"temporal_lidar_v3" and frames_text == b"4") or (contract == CONTRACT and frames_text in (b"4", b"8")):
        frames = int(frames_text)
    else:
        raise ValueError("predictor_contract_or_frame_count_mismatch")
    with torch.inference_mode():
        result = model(torch.zeros(1, 2, frames, 128, dtype=torch.float32))
    if not isinstance(result, torch.Tensor) or tuple(result.shape) != (1, 128, 2):
        raise ValueError("predictor_output_shape")
    if not torch.isfinite(result).all().item():
        raise ValueError("predictor_nonfinite_output")
    return model, frames, contract.decode("ascii")


class CbfVelocityPredictor(Node):
    def __init__(self):
        super().__init__("cbf_velocity_predictor")
        self.declare_parameter("cbf_mode", "static")
        self.declare_parameter("velocity_predictor_jit_path", "")
        self.declare_parameter("predictor_warning_threshold_ms", 50.0)
        self.mode = self.get_parameter("cbf_mode").value
        if self.mode not in ("static", "dynamic"):
            raise ValueError("cbf_mode must be static or dynamic")
        self._warning_threshold_ms = float(self.get_parameter("predictor_warning_threshold_ms").value)
        if not np.isfinite(self._warning_threshold_ms) or self._warning_threshold_ms <= 0:
            raise ValueError("predictor_warning_threshold_ms must be a finite positive number")
        self._condition = Condition()
        self._history = ScanHistory()
        self._session = None
        self._pending = None
        self._running = False
        self._model = None
        self._frames = None
        self._inference_count = 0
        self._inference_total_ms = 0.0
        self._inference_max_ms = 0.0
        self._last_latency_log = monotonic()
        if self.mode == "static":
            self.get_logger().info("Static CBF selected; predictor is idle.")
            return
        path = self.get_parameter("velocity_predictor_jit_path").value
        try:
            started = monotonic()
            self._model, self._frames, contract = load_predictor(path)
        except Exception as error:
            self.get_logger().error(f"Dynamic predictor unavailable; CBF will use static fallback: {error}")
            return
        self.get_logger().info(
            f"Dynamic predictor loaded: path={path}, contract={contract}, frames={self._frames}, "
            f"load_and_probe_ms={(monotonic() - started) * 1000:.1f}"
        )
        self._publisher = self.create_publisher(CbfDynamicSnapshot, "/cbf/dynamic_snapshot", 1)
        self.create_subscription(CbfWorldScan, "/cbf/world_scan", self._scan_callback, 10)
        self.create_subscription(CbfControlSnapshot, "/cbf/snapshot", self._snapshot_callback, qos_profile_sensor_data)
        self._running = True
        self._worker = Thread(target=self._run, daemon=True)
        self._worker.start()

    def destroy_node(self):
        with self._condition:
            self._running = False
            self._condition.notify_all()
        if hasattr(self, "_worker"):
            self._worker.join(timeout=1.0)
        return super().destroy_node()

    def _add_scan(self, scan):
        self._history.add(scan)

    def _scan_callback(self, scan):
        with self._condition:
            if self._session is None:
                self._session = scan.session_id
            if scan.session_id != self._session:
                return  # Only an evaluated snapshot can begin a new session.
            self._add_scan(scan)

    def _snapshot_callback(self, snapshot):
        if not snapshot.valid:
            return
        with self._condition:
            if snapshot.session_id != self._session:
                self._session = snapshot.session_id
                self._history.clear()
                self._pending = None
            # The embedded capture handles ROS delivery order between the two topics.
            if snapshot.scan.session_id == self._session:
                self._history.add(snapshot.scan, replace=True)
            self._pending = snapshot
            self._condition.notify()

    def _run(self):
        import torch

        while True:
            with self._condition:
                while self._running and self._pending is None:
                    self._condition.wait()
                if not self._running:
                    return
                snapshot = self._pending
                self._pending = None
                capture_time = stamp_ns(snapshot.scan.header.stamp)
                scans = self._history.select(snapshot.scan, self._frames)
            started = monotonic()
            inference_ms = None
            try:
                if snapshot.scan.session_id != snapshot.session_id:
                    raise ValueError("predictor_scan_session_mismatch")
                if self._frames == 8 and len(scans) < self._frames:
                    raise ValueError("insufficient_predictor_history")
                evaluation_time = stamp_ns(snapshot.header.stamp)
                age_s = (evaluation_time - capture_time) * 1e-9
                if age_s < 0:
                    raise ValueError("capture_after_evaluation")
                hits, states = history_arrays(scans, self._frames)
                lidar, winner = project_history(hits, states, snapshot.robot_xy_m, snapshot.robot_yaw)
                inference_started = monotonic()
                with torch.inference_mode():
                    prediction = self._model(torch.from_numpy(lidar).unsqueeze(0))
                inference_ms = (monotonic() - inference_started) * 1000.0
                if not isinstance(prediction, torch.Tensor) or tuple(prediction.shape) != (1, 128, 2):
                    raise ValueError("predictor_output_shape")
                points, velocity, reflected = prepare_points(
                    hits[0], winner, prediction[0].detach().cpu().numpy(), snapshot.robot_yaw, age_s
                )
                message = CbfDynamicSnapshot()
                message.base = snapshot
                message.valid = True
                message.reason = "matched_prediction"
                message.front_points_xy_m = points.ravel().tolist()
                message.front_velocity_xy_mps = velocity.ravel().tolist()
                message.reflection_hits = reflected.tolist()
                with self._condition:
                    if self._running and self._session == snapshot.session_id:
                        self._publisher.publish(message)
            except Exception as error:
                if str(error) != "insufficient_predictor_history":
                    self.get_logger().error(f"Dynamic prediction rejected; static fallback remains active: {error}")
                message = CbfDynamicSnapshot()
                message.base = snapshot
                message.valid = False
                message.reason = str(error)
                with self._condition:
                    if self._running and self._session == snapshot.session_id:
                        self._publisher.publish(message)
            elapsed = monotonic() - started
            if inference_ms is not None:
                self._inference_count += 1
                self._inference_total_ms += inference_ms
                self._inference_max_ms = max(self._inference_max_ms, inference_ms)
                if monotonic() - self._last_latency_log >= 10.0:
                    self.get_logger().info(
                        f"Dynamic predictor frames={self._frames}, inference_count={self._inference_count}, "
                        f"inference_mean_ms={self._inference_total_ms / self._inference_count:.1f}, "
                        f"inference_max_ms={self._inference_max_ms:.1f}"
                    )
                    self._inference_count = 0
                    self._inference_total_ms = 0.0
                    self._inference_max_ms = 0.0
                    self._last_latency_log = monotonic()
            self.get_logger().debug(
                f"Dynamic predictor frames={self._frames}, inference_ms={inference_ms}, processing_ms={elapsed * 1000:.1f}"
            )
            if elapsed * 1000.0 > self._warning_threshold_ms:
                self.get_logger().warn(f"Dynamic predictor took {elapsed * 1000:.1f} ms; latest snapshot may be stale.")


def main():
    rclpy.init()
    node = CbfVelocityPredictor()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
