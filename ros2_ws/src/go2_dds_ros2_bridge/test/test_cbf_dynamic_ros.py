"""Isolated predictor-to-controller test; never publish on a robot ROS domain."""

import os
import subprocess
import time
import math

import pytest

rclpy = pytest.importorskip("rclpy")
torch = pytest.importorskip("torch")
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import TwistStamped
from diagnostic_msgs.msg import DiagnosticArray
from go2_dds_ros2_bridge_msgs.msg import CbfControlSnapshot, CbfDynamicSnapshot, CbfWorldScan
from go2_dds_ros2_bridge.nodes.cbf_velocity_predictor_node import CbfVelocityPredictor, load_predictor


class ConstantVelocity(torch.nn.Module):
    def forward(self, lidar):
        output = torch.zeros((lidar.shape[0], 128, 2), dtype=lidar.dtype)
        output[:, :, 0] = -0.5
        return output


class InvalidVelocity(torch.nn.Module):
    def forward(self, lidar):
        return torch.full((lidar.shape[0], 128, 2), float("nan"))


def _model(path, contract=b"temporal_lidar_v4", frames=b"4"):
    traced = torch.jit.trace(ConstantVelocity(), torch.zeros(1, 2, int(frames), 128))
    torch.jit.save(traced, str(path), _extra_files={
        "projection_contract.txt": contract, "num_frames.txt": frames,
    })


def _status_values(message):
    return {entry.key: entry.value for entry in message.status[0].values}


def test_model_metadata_is_checked(tmp_path):
    path = tmp_path / "model.pt"
    _model(path, contract=b"temporal_lidar_v3")
    model, frames, contract = load_predictor(str(path))
    assert (frames, contract) == (4, "temporal_lidar_v3")
    assert tuple(model(torch.zeros(1, 2, 4, 128)).shape) == (1, 128, 2)
    _model(path, contract=b"temporal_lidar_v2")
    with pytest.raises(ValueError, match="contract_or_frame_count"):
        load_predictor(str(path))
    _model(path, contract=b"temporal_lidar_v3", frames=b"8")
    with pytest.raises(ValueError, match="contract_or_frame_count"):
        load_predictor(str(path))
    _model(path, frames=b"8")
    model, frames, contract = load_predictor(str(path))
    assert (frames, contract) == (8, "temporal_lidar_v4")
    assert tuple(model(torch.zeros(1, 2, 8, 128)).shape) == (1, 128, 2)
    _model(path, frames=b"6")
    with pytest.raises(ValueError, match="contract_or_frame_count"):
        load_predictor(str(path))
    _model(path, contract=b"")
    with pytest.raises(ValueError, match="contract_or_frame_count"):
        load_predictor(str(path))
    traced = torch.jit.trace(ConstantVelocity(), torch.zeros(1, 2, 4, 128))
    torch.jit.save(traced, str(path))
    with pytest.raises(ValueError, match="contract_or_frame_count"):
        load_predictor(str(path))


def test_model_nonfinite_probe_is_rejected(tmp_path):
    path = tmp_path / "bad.pt"
    traced = torch.jit.trace(InvalidVelocity(), torch.zeros(1, 2, 8, 128))
    torch.jit.save(traced, str(path), _extra_files={
        "projection_contract.txt": b"temporal_lidar_v4", "num_frames.txt": b"8",
    })
    with pytest.raises(ValueError, match="predictor_nonfinite_output"):
        load_predictor(str(path))


@pytest.mark.parametrize("frames", [4, 8])
def test_dynamic_selection_then_static_fallback(tmp_path, frames):
    assert int(os.environ.get("ROS_DOMAIN_ID", "0")) != 0
    model_path = tmp_path / "model.pt"
    _model(model_path, frames=str(frames).encode())
    rclpy.init(args=["--ros-args", "-p", "cbf_mode:=dynamic",
                     "-p", f"velocity_predictor_jit_path:={model_path}"])
    predictor = CbfVelocityPredictor()
    assert predictor._frames == frames
    harness = Node("cbf_dynamic_test_harness")
    executor = SingleThreadedExecutor()
    executor.add_node(predictor)
    executor.add_node(harness)
    snapshot_pub = harness.create_publisher(CbfControlSnapshot, "/cbf/snapshot", 10)
    policy_pub = harness.create_publisher(TwistStamped, "/policy_vel", 10)
    diagnostics = []
    commands = []
    harness.create_subscription(DiagnosticArray, "/cbf/status", diagnostics.append, 10)
    harness.create_subscription(TwistStamped, "/cbf/test_command", commands.append, 10)
    from ament_index_python.packages import get_package_prefix
    executable = get_package_prefix("go2_cbf_control") + "/lib/go2_cbf_control/cbf_control"
    log = open(tmp_path / "controller.log", "w")
    process = subprocess.Popen([
        executable, "--ros-args", "-r", "/cmd_vel:=/cbf/test_command",
        "-p", "cbf_mode:=dynamic", "-p", "scan_timeout_s:=1.0",
        "-p", "state_timeout_s:=0.2", "-p", "publish_deadline_s:=0.04",
        "-p", "solver_time_limit_s:=0.02", "-p", "planar_deadzone_mps:=0.0",
    ], stdout=log, stderr=log)
    sequence = 0
    scan_stamp = 0
    session_id = "test_session"
    predictor_alive = True

    def cycle():
        nonlocal sequence, scan_stamp
        now_ns = harness.get_clock().now().nanoseconds
        if not scan_stamp or now_ns - scan_stamp > 150_000_000:
            sequence += 1
            scan_stamp = now_ns - 50_000_000
        scan = CbfWorldScan()
        scan.header.frame_id = "cbf_world"
        scan.header.stamp = Time(nanoseconds=scan_stamp).to_msg()
        scan.scan_start = Time(nanoseconds=scan_stamp - 50_000_000).to_msg()
        scan.session_id = session_id
        scan.sequence = sequence
        scan.points_xy_m[128:130] = [1.0, 0.0]
        scan.hits[64] = 1
        scan.capture_points_xy_m[:2] = [1.0, 0.0]
        scan.ray_states[0] = 2
        snapshot = CbfControlSnapshot()
        snapshot.header.frame_id = "cbf_world"
        snapshot.header.stamp = Time(nanoseconds=now_ns - 10_000_000).to_msg()
        snapshot.valid = True
        snapshot.session_id = session_id
        snapshot.robot_xy_m = [0.0, 0.0]
        snapshot.robot_yaw = 0.0
        snapshot.velocity_xy_mps = [0.0, 0.0]
        snapshot.scan = scan
        snapshot_pub.publish(snapshot)
        policy = TwistStamped()
        policy.header.frame_id = "base_link"
        policy.header.stamp = Time(nanoseconds=now_ns).to_msg()
        policy_pub.publish(policy)
        executor.spin_once(timeout_sec=0.01)
        return snapshot

    try:
        if frames == 8:
            warmup_until = time.monotonic() + 0.8
            while time.monotonic() < warmup_until and not any(
                "insufficient_predictor_history" in _status_values(d).get("dynamic_reason", "")
                for d in diagnostics
            ):
                cycle()
            assert any("insufficient_predictor_history" in _status_values(d).get("dynamic_reason", "")
                       for d in diagnostics)
            assert not any(_status_values(d).get("dynamic_active") == "1.000000" for d in diagnostics)
        until = time.monotonic() + (3.0 if frames == 8 else 2.0)
        while time.monotonic() < until and not any(
            _status_values(d).get("dynamic_active") == "1.000000" for d in diagnostics
        ):
            cycle()
        assert process.poll() is None, (tmp_path / "controller.log").read_text()
        assert any(_status_values(d).get("dynamic_active") == "1.000000" for d in diagnostics)
        assert any(c.twist.linear.x < -0.1 for c in commands)
        assert len(harness.get_publishers_info_by_topic("/cbf/test_command")) == 1

        if frames == 8:
            sequence += 1  # Skip one physical capture sequence.
            scan_stamp = 0
            prior = len(diagnostics)
            until = time.monotonic() + 1.0
            while time.monotonic() < until and not any(
                "insufficient_predictor_history" in _status_values(d).get("dynamic_reason", "")
                for d in diagnostics[prior:]
            ):
                cycle()
            assert any("insufficient_predictor_history" in _status_values(d).get("dynamic_reason", "")
                       for d in diagnostics[prior:])
            prior = len(diagnostics)
            until = time.monotonic() + 2.5
            while time.monotonic() < until and not any(
                _status_values(d).get("dynamic_active") == "1.000000" for d in diagnostics[prior:]
            ):
                cycle()
            assert any(_status_values(d).get("dynamic_active") == "1.000000" for d in diagnostics[prior:])

        healthy_model = predictor._model
        predictor._model = InvalidVelocity()
        until = time.monotonic() + 1.0
        while time.monotonic() < until and not any(
            "dynamic_preparation_failed" in _status_values(d).get("dynamic_reason", "")
            for d in diagnostics[-5:]
        ):
            cycle()
        assert any("dynamic_preparation_failed" in _status_values(d).get("dynamic_reason", "")
                   for d in diagnostics[-5:])
        assert any("static_cbf_fallback" in d.status[0].message for d in diagnostics[-5:])
        assert any(_status_values(d).get("dynamic_active") == "0.000000" for d in diagnostics[-5:])
        assert commands and math.isfinite(commands[-1].twist.linear.x)
        predictor._model = healthy_model
        prior = len(diagnostics)
        until = time.monotonic() + 1.0
        while time.monotonic() < until and not any(
            _status_values(d).get("dynamic_active") == "1.000000" for d in diagnostics[prior:]
        ):
            cycle()
        assert any(_status_values(d).get("dynamic_active") == "1.000000" for d in diagnostics[prior:])

        session_id = "reset_session"
        prior = len(diagnostics)
        until = time.monotonic() + (2.5 if frames == 8 else 1.0)
        while time.monotonic() < until and not any(
            _status_values(d).get("world_session") == session_id and
            _status_values(d).get("dynamic_active") == "1.000000"
            for d in diagnostics[prior:]
        ):
            cycle()
        assert any(_status_values(d).get("world_session") == session_id and
                   _status_values(d).get("dynamic_active") == "1.000000"
                   for d in diagnostics[prior:])

        executor.remove_node(predictor)
        predictor.destroy_node()
        predictor_alive = False
        until = time.monotonic() + 0.5
        while time.monotonic() < until:
            cycle()
        assert any("static_cbf_fallback" in d.status[0].message for d in diagnostics[-10:])

        dynamic_pub = harness.create_publisher(CbfDynamicSnapshot, "/cbf/dynamic_snapshot", 10)
        bad = CbfDynamicSnapshot()
        bad.valid = True
        bad.base = cycle()
        bad.base.scan.sequence += 100
        dynamic_pub.publish(bad)
        until = time.monotonic() + 0.5
        while time.monotonic() < until and not any(
            "dynamic_capture_or_session_mismatch" in _status_values(d).get("dynamic_reason", "")
            for d in diagnostics[-5:]
        ):
            cycle()
        assert any("dynamic_capture_or_session_mismatch" in _status_values(d).get("dynamic_reason", "")
                   for d in diagnostics[-5:])
    finally:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait()
        if predictor_alive:
            predictor.destroy_node()
        executor.remove_node(harness)
        harness.destroy_node()
        executor.shutdown()
        rclpy.shutdown()
        log.close()
