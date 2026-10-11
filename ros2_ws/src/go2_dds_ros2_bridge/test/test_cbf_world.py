import math
import numpy as np
import pytest
from go2_dds_ros2_bridge.cbf_world import pose_at, slerp, aligned_state, world_points, bracket
from go2_dds_ros2_bridge.temporal_lidar_processing import cbf_xyz_bins


def q(yaw):
    return np.array([0., 0., math.sin(yaw/2), math.cos(yaw/2)])


@pytest.mark.parametrize('yaw', [0, math.pi/2, -math.pi/2, math.pi, .37])
@pytest.mark.parametrize('shift', [[0,0,0], [2,-3,1]])
def test_full_transform_roundtrip(yaw, shift):
    from scipy.spatial.transform import Rotation
    r = Rotation.from_euler('xyz', [.3, -.2, yaw]).as_matrix()
    wr = Rotation.from_euler('xyz', [-.1, .2, .7]).as_matrix()
    p, t, offset = np.array(shift), np.array([1.,2.,3.]), np.array([.03,0,-.04])
    points = np.array([[1,2,.6], [-1,-2,-.2]])
    result = world_points(points, p, r, wr, t, offset)
    # Independent homogeneous-matrix oracle including the body/base lever arm.
    a, b, c = np.eye(4), np.eye(4), np.eye(4)
    a[:3,:3], a[:3,3] = wr, t
    b[:3,:3], b[:3,3] = r, p
    c[:3,3] = offset
    expected = (a @ b @ c @ np.c_[points, np.ones(2)].T).T[:,:3]
    np.testing.assert_allclose(result, expected, atol=1e-8)
    back = (np.linalg.inv(a @ b @ c) @ np.c_[result,np.ones(2)].T).T[:,:3]
    np.testing.assert_allclose(back, points, atol=1e-8)


def test_yaw_wrap_and_quaternion_sign():
    h = [(0,np.zeros(3),q(math.radians(179))), (100_000_000,np.ones(3),q(math.radians(-179)))]
    p,r,gap = pose_at(h, 50_000_000)
    np.testing.assert_allclose(p, .5)
    np.testing.assert_allclose(r[:2,:2], -np.eye(2), atol=1e-8)
    np.testing.assert_allclose(slerp(q(.4), -q(.4), .5), q(.4), atol=1e-8)
    assert gap == .1


def test_gap_is_not_scan_age():
    h = [(0,np.zeros(3),q(0)),(101_000_000,np.ones(3),q(0))]
    with pytest.raises(ValueError, match='interpolation_gap'): pose_at(h,50_000_000)
    pose_at(h,101_000_000) # exact sample does not interpolate across a gap
    with pytest.raises(ValueError, match='not_bracketed'): pose_at(h,102_000_000)


@pytest.mark.parametrize('velocity', [[1,0,0],[-1,0,0],[0,1,0]])
def test_held_static_obstacle_and_measured_state(velocity):
    v = np.array(velocity, float)
    h = [(i*50_000_000, v*i*.05, q(0)) for i in range(8)]
    vel = [(item[0],v) for item in h]
    fixed = np.array([1.,0.,0.])
    for i in range(1,8):
        stamp,p,yaw,wv,gap=aligned_state(h[:i+1],vel[:i+1],np.eye(3),np.zeros(3),np.zeros(3),i*50_000_000+30_000_000)
        np.testing.assert_allclose(fixed-p, fixed-v*i*.05,atol=1e-8)
        np.testing.assert_allclose(wv,v)
    with pytest.raises(ValueError,match='stale'):
        aligned_state(h,vel,np.eye(3),np.zeros(3),np.zeros(3),500_000_000)


def test_velocity_rotated_at_its_own_attitude():
    h=[(0,np.zeros(3),q(0)),(100_000_000,np.zeros(3),q(math.pi/2))]
    velocities=[(0,np.array([1.,0,0])),(100_000_000,np.array([0.,-1,0]))]
    _,_,_,v,_=aligned_state(h,velocities,np.eye(3),np.zeros(3),np.zeros(3),50_000_000)
    np.testing.assert_allclose(v,[1,0,0],atol=1e-8)


def test_xyz_reducer_preserves_real_points_and_identity():
    points=np.zeros((256,3));states=np.zeros(256)
    points[4]=[1,0,.7];points[5]=[2,0,.8];states[4:6]=2
    rear=np.array([[-1,0,.4],[-2,0,.9]])
    f,h,r,rh=cbf_xyz_bins(points,states,rear)
    assert h.sum()==1 and h[2]==1
    np.testing.assert_array_equal(f[2],[1,0,.7])
    assert rh.sum()==1
    assert any(np.array_equal(r[rh==1][0],p) for p in rear)


@pytest.mark.parametrize('yaw',[0.,.4,1.57,-2.])
def test_common_world_transform_preserves_relative_geometry(yaw):
    from scipy.spatial.transform import Rotation
    r=Rotation.from_euler('z',yaw).as_matrix()
    robot=np.array([.4,-.3,.2]);points=np.array([[2.,1.,.7],[-1.,2.,.2]])
    translation=np.array([100.,-80.,5.])
    relative=(points@r.T+translation)-(robot@r.T+translation)
    np.testing.assert_allclose(relative,(points-robot)@r.T,atol=1e-8)
    np.testing.assert_allclose(np.linalg.norm(relative[:,:2],axis=1),np.linalg.norm((points-robot)[:,:2],axis=1),atol=1e-8)


def test_interpolation_endpoints_and_invalid_quaternions():
    h=[(0,np.zeros(3),q(0)),(100_000_000,np.ones(3),q(.5))]
    assert bracket(h,0)[2]==0
    assert bracket(h,100_000_000)[2]==0
    with pytest.raises(ValueError,match='not_bracketed'):bracket(h,-1)
    with pytest.raises(ValueError,match='invalid_quaternion'):slerp(np.zeros(4),q(0),.5)
