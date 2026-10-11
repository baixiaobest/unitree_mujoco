#!/usr/bin/env python3
"""Prepare matched four-frame obstacle velocities outside the CBF control timer."""

from collections import deque
from threading import Condition, Thread
from time import monotonic

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from go2_dds_ros2_bridge.cbf_dynamic import CONTRACT, FRAMES, history_arrays, prepare_points, project_history
from go2_dds_ros2_bridge_msgs.msg import CbfControlSnapshot, CbfDynamicSnapshot, CbfWorldScan


def stamp_ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def load_predictor(path):
    """Reject a model whose projection or temporal shape is not this ROS path."""
    import torch

    metadata = {"projection_contract.txt": b"", "num_frames.txt": b""}
    model = torch.jit.load(path, map_location="cpu", _extra_files=metadata).eval()
    if metadata["projection_contract.txt"] != CONTRACT or metadata["num_frames.txt"] != b"4":
        raise ValueError("predictor_contract_or_frame_count_mismatch")
    with torch.inference_mode():
        result = model(torch.zeros(1, 2, FRAMES, 128, dtype=torch.float32))
    if not isinstance(result, torch.Tensor) or tuple(result.shape) != (1, 128, 2):
        raise ValueError("predictor_output_shape")
    return model


class CbfVelocityPredictor(Node):
    def __init__(self):
        super().__init__("cbf_velocity_predictor")
        self.declare_parameter("cbf_mode", "static")
        self.declare_parameter("velocity_predictor_jit_path", "")
        self.mode = self.get_parameter("cbf_mode").value
        if self.mode not in ("static", "dynamic"):
            raise ValueError("cbf_mode must be static or dynamic")
        self._condition = Condition()
        self._history = deque(maxlen=FRAMES)
        self._session = None
        self._pending = None
        self._running = False
        self._model = None
        if self.mode == "static":
            self.get_logger().info("Static CBF selected; predictor is idle.")
            return
        path = self.get_parameter("velocity_predictor_jit_path").value
        try:
            self._model = load_predictor(path)
        except Exception as error:
            self.get_logger().error(f"Dynamic predictor unavailable; CBF will use static fallback: {error}")
            return
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
        identity = (stamp_ns(scan.header.stamp), scan.sequence)
        if self._history:
            newest = self._history[0]
            last = (stamp_ns(newest.header.stamp), newest.sequence)
            if identity == last:
                return
            if identity[0] <= last[0]:
                return
            if scan.sequence != ((newest.sequence + 1) & 0xFFFFFFFF):
                self._history.clear()  # Missing scans invalidate old temporal neighbors.
        self._history.appendleft(scan)

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
            self._add_scan(snapshot.scan)
            if snapshot.session_id != self._session:
                return
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
                scans = [s for s in self._history if stamp_ns(s.header.stamp) <= capture_time]
                if not scans or scans[0].sequence != snapshot.scan.sequence or scans[0].session_id != snapshot.session_id:
                    continue
                scans = scans[:FRAMES]
            started = monotonic()
            try:
                evaluation_time = stamp_ns(snapshot.header.stamp)
                age_s = (evaluation_time - capture_time) * 1e-9
                if age_s < 0:
                    raise ValueError("capture_after_evaluation")
                hits, states = history_arrays(scans)
                lidar, winner = project_history(hits, states, snapshot.robot_xy_m, snapshot.robot_yaw)
                with torch.inference_mode():
                    prediction = self._model(torch.from_numpy(lidar).unsqueeze(0))
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
                self.get_logger().error(f"Dynamic prediction rejected; static fallback remains active: {error}")
                message = CbfDynamicSnapshot()
                message.base = snapshot
                message.valid = False
                message.reason = str(error)
                with self._condition:
                    if self._running and self._session == snapshot.session_id:
                        self._publisher.publish(message)
            elapsed = monotonic() - started
            if elapsed > 0.05:
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
