from types import SimpleNamespace

from go2_dds_ros2_bridge.nodes.imu_bridge_node import GO2_JOINT_NAMES, joint_state_from_lowstate


def lowstate():
    return SimpleNamespace(motor_state=[
        SimpleNamespace(q=float(index), dq=float(index) / 10, tau_est=float(index) / 100)
        for index in range(12)
    ])


def test_unitree_motor_order_matches_urdf_joint_names():
    msg = joint_state_from_lowstate(lowstate(), 1_234_567_890)
    assert msg is not None
    assert msg.name == list(GO2_JOINT_NAMES)
    assert msg.name[:3] == ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"]
    assert msg.name[9:] == ["RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"]
    assert list(msg.position) == list(map(float, range(12)))
    assert msg.header.stamp.sec == 1
    assert msg.header.stamp.nanosec == 234_567_890


def test_invalid_positions_do_not_publish_joint_state():
    sample = lowstate()
    sample.motor_state[4].q = float("nan")
    assert joint_state_from_lowstate(sample, 0) is None
    sample.motor_state = sample.motor_state[:11]
    assert joint_state_from_lowstate(sample, 0) is None
