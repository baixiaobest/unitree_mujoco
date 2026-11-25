import math

from go2_dds_ros2_bridge.learned_motion_integration import LearnedMotionIntegrator


def test_straight_line_and_duplicate_tick():
    odom = LearnedMotionIntegrator()
    assert odom.update(1_000_000_000, 0, (1, 0, 0), (0, 0, 0))
    assert odom.update(2_000_000_000, 1, (1, 0, 0), (0, 0, 0))
    # Long gaps do not integrate stale velocity.
    assert odom.position == (0, 0, 0)
    assert odom.update(2_100_000_000, 2, (1, 0, 0), (0, 0, 0))
    assert math.isclose(odom.position[0], 0.1)
    assert odom.update(2_100_000_000, 3, (1, 0, 0), (0, 0, 0))
    assert math.isclose(odom.position[0], 0.1)
    assert not odom.update(2_050_000_000, 4, (1, 0, 0), (0, 0, 0))


def test_constant_turn():
    odom = LearnedMotionIntegrator()
    odom.update(0, 0, (1, 0, 0), (0, 0, 1))
    odom.update(100_000_000, 1, (1, 0, 0), (0, 0, 1))
    assert math.isclose(odom.orientation[0], math.cos(0.05), abs_tol=1e-6)
    assert math.isclose(odom.orientation[3], math.sin(0.05), abs_tol=1e-6)
    assert odom.position[0] > 0.09
    assert odom.position[1] > 0
