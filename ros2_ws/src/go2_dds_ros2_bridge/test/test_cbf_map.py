import numpy as np

from go2_dds_ros2_bridge.cbf_map import floor_and_rear, load_pcd_xyz, voxel_downsample, supported_obstacles


def test_floor_and_rear_rejects_ground_and_front():
    xs, ys = np.meshgrid(np.linspace(-2, 2, 41), np.linspace(-2, 2, 41))
    floor = np.column_stack((xs.ravel(), ys.ravel(), 0.05 * xs.ravel() - 0.03 * ys.ravel() - 0.3))
    rear = np.array([[-1.0, 0.0, -0.3 + 0.05 * -1.0 + 0.5]])
    front = np.array([[1.0, 0.0, -0.3 + 0.05 + 0.5]])
    result = floor_and_rear(np.vstack((floor, rear, front)), min_support_points=1)
    assert result is not None
    support, candidates, plane, _, _ = result
    assert len(support) > 100
    np.testing.assert_allclose(plane, [0.05, -0.03, -0.3], atol=0.01)
    assert len(candidates) == 1
    np.testing.assert_allclose(candidates[0], rear[0])


def test_no_floor_does_not_make_obstacles():
    wall = np.column_stack((np.full(80, -1.0), np.linspace(-1, 1, 80), np.linspace(0, 1, 80)))
    assert floor_and_rear(wall) is None


def test_map_obstacle_range_is_independent():
    xs, ys = np.meshgrid(np.linspace(-2, 2, 41), np.linspace(-2, 2, 41))
    floor = np.column_stack((xs.ravel(), ys.ravel(), np.full(xs.size, -0.3)))
    near = np.array([[-2.0, 0.0, 0.2]])
    far = np.array([[-4.0, 0.0, 0.2]])
    _, candidates, _, _, _ = floor_and_rear(np.vstack((floor, near, far)), max_range=3.0,
                                           min_support_points=1)
    assert len(candidates) == 1
    np.testing.assert_allclose(candidates[0], near[0])


def test_binary_pcd_xyz(tmp_path):
    path = tmp_path / 'map.pcd'
    points = np.array([(1., 2., 3., 4.), (-1., 0., 0.5, 1.)],
                      dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('intensity', '<f4')])
    path.write_bytes(b'\n'.join((b'VERSION 0.7', b'FIELDS x y z intensity', b'SIZE 4 4 4 4',
                                b'TYPE F F F F', b'COUNT 1 1 1 1', b'WIDTH 2', b'HEIGHT 1',
                                b'POINTS 2', b'DATA binary')) + b'\n' + points.tobytes())
    np.testing.assert_allclose(load_pcd_xyz(path), [[1, 2, 3], [-1, 0, 0.5]])


def test_voxel_downsample_preserves_separate_heights():
    points = np.array([[0.01, 0, 0], [0.02, 0, 0.01], [0.01, 0, 0.3]])
    result = voxel_downsample(points, 0.1)
    assert len(result) == 2
    assert np.max(result[:, 2]) == 0.3


def test_sparse_map_returns_are_rejected_before_binning():
    points = np.array([[-1.0, 0, 0.3], [-1.08, 0, 0.32], [-0.95, 0.05, 0.31],
                       [-2.0, 0, 0.3], [-2.05, 0, 0.3]])
    accepted, rejected = supported_obstacles(points, radius_m=0.2, min_points=3)
    assert len(accepted) == 3
    assert len(rejected) == 2
    np.testing.assert_allclose(accepted, points[:3])
