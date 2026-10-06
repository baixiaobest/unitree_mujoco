"""Headless Go2 kinematic regression against MuJoCo ground-truth transforms.

This validates sampled geometry under prescribed trajectories, not closed-loop
locomotion or physical stopping performance. No DDS/robot connection is used.
"""
from pathlib import Path
import math
import numpy as np
import pytest
from go2_dds_ros2_bridge.cbf_world import world_points, pose_at


@pytest.mark.parametrize('motion', ['stationary','approach','retreat','lateral','rotate','combined'])
def test_go2_held_scans(motion):
    mujoco = pytest.importorskip("mujoco")
    root=Path(__file__).resolve().parents[4]
    model=mujoco.MjModel.from_xml_path(str(root/'unitree_robots/go2/scene.xml'))
    data=mujoco.MjData(model)
    base=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'base_link')
    assert base>=0
    obstacles=np.array([[2.,0.,.7],[-2.,0.,.4],[0.,2.,.8]])
    held=None
    max_error=0.
    for step in range(250):
        t=step*.02
        x=.15*t if motion in ('approach','combined') else (-.15*t if motion=='retreat' else 0.)
        y=.1*t if motion in ('lateral','combined') else 0.
        yaw=.4*t if motion in ('rotate','combined') else 0.
        # Include small roll/pitch, so flattening before transformation fails.
        quat=np.zeros(4)
        mujoco.mju_euler2Quat(quat,np.array([.08*math.sin(t),.05*math.cos(t),yaw]),'xyz')
        data.qpos[:3]=[x,y,.35];data.qpos[3:7]=quat
        mujoco.mj_forward(model,data)
        position=data.xpos[base].copy();rotation=data.xmat[base].reshape(3,3).copy()
        offset=np.array([.03,0,-.04])
        if step%8==0:
            points=(obstacles-position)@rotation-offset
            held=world_points(points,position,rotation,np.eye(3),np.zeros(3),offset)
        # An independently evaluated MuJoCo current pose supplies the reference.
        expected=(obstacles-position)@rotation-offset
        current=(held-position)@rotation-offset
        error=np.max(np.abs(current-expected))
        max_error=max(max_error,float(error))
        assert error<1e-8
    print(f'{motion}: 250 Go2 poses; maximum held-scan transform error={max_error:.3g} m')
