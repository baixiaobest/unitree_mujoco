"""Isolated ROS integration test. Run with ROS_DOMAIN_ID=187 and built overlay sourced."""
import math
import os
import subprocess
import time
import numpy as np
import pytest
rclpy = pytest.importorskip('rclpy')
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.time import Time
from geometry_msgs.msg import TransformStamped, TwistStamped
from nav_msgs.msg import Odometry
from diagnostic_msgs.msg import DiagnosticArray
from std_srvs.srv import Trigger
from tf2_ros import StaticTransformBroadcaster
from go2_dds_ros2_bridge_msgs.msg import CbfScan3D, CbfControlSnapshot
from go2_dds_ros2_bridge.nodes.cbf_world_node import CbfWorldNode, ns
from go2_dds_ros2_bridge.cbf_world import pose_at


def test_live_world_pipeline_and_controller(tmp_path):
    executable = os.environ.get('CBF_TEST_EXECUTABLE')
    if not executable:
        from ament_index_python.packages import get_package_prefix
        executable = get_package_prefix('go2_cbf_control') + '/lib/go2_cbf_control/cbf_control'
    assert int(os.environ.get('ROS_DOMAIN_ID', '0')) != 0, 'Use an isolated ROS domain'
    rclpy.init()
    world = CbfWorldNode()
    harness = Node('cbf_world_test_harness')
    executor = SingleThreadedExecutor()
    executor.add_node(world); executor.add_node(harness)
    log = open(tmp_path/'controller.log', 'w')
    process = subprocess.Popen([executable, '--ros-args', '-r', '/cmd_vel:=/cbf/test_command',
                                '-p','enable_navigation_slew:=false'], stdout=log, stderr=log)
    tf = StaticTransformBroadcaster(harness)
    transforms = []
    for parent,child,xyz in [('camera_init_correct','camera_init',[.7,-.2,.1]),('body','base_link',[.03,0,-.04])]:
        t=TransformStamped(); t.header.frame_id=parent; t.child_frame_id=child
        t.header.stamp=harness.get_clock().now().to_msg()
        t.transform.rotation.w=1.
        t.transform.translation.x,t.transform.translation.y,t.transform.translation.z=map(float,xyz)
        transforms.append(t)
    tf.sendTransform(transforms)
    pose_pub=harness.create_publisher(Odometry,'/Odometry',10)
    velocity_pub=harness.create_publisher(TwistStamped,'/estimated_velocity',10)
    scan_pub=harness.create_publisher(CbfScan3D,'/cbf/scan_xyz',10)
    policy_pub=harness.create_publisher(TwistStamped,'/policy_vel',10)
    snapshots=[];commands=[];diagnostics=[]
    harness.create_subscription(CbfControlSnapshot,'/cbf/snapshot',snapshots.append,100)
    harness.create_subscription(TwistStamped,'/cbf/test_command',commands.append,100)
    harness.create_subscription(DiagnosticArray,'/cbf/status',diagnostics.append,100)
    origin=harness.get_clock().now().nanoseconds
    history=[]; sequence=0
    fixed=np.array([2.,1.,.7]) # fixed in camera_init
    last_scan=0

    def cycle(publish_state=True,publish_scan=True):
        nonlocal sequence,last_scan
        now=harness.get_clock().now().nanoseconds
        stamp=now-20_000_000
        elapsed=(stamp-origin)*1e-9
        yaw=.3*elapsed
        c,s=math.cos(yaw),math.sin(yaw)
        r=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
        p=np.array([.2*elapsed,.1*elapsed,0])
        if publish_state:
            msg=Odometry();msg.header.frame_id='camera_init';msg.child_frame_id='body'
            msg.header.stamp=Time(nanoseconds=stamp).to_msg()
            msg.pose.pose.position.x,msg.pose.pose.position.y,msg.pose.pose.position.z=map(float,p)
            msg.pose.pose.orientation.z=math.sin(yaw/2);msg.pose.pose.orientation.w=math.cos(yaw/2)
            pose_pub.publish(msg)
            history.append((stamp,p,np.array([0,0,math.sin(yaw/2),math.cos(yaw/2)])))
            v=TwistStamped();v.header=msg.header;v.header.frame_id='base_link'
            vb=r.T@np.array([.2,.1,0])
            v.twist.linear.x,v.twist.linear.y,v.twist.linear.z=map(float,vb)
            velocity_pub.publish(v)
            if publish_scan and now-last_scan>160_000_000:
                raw=CbfScan3D();raw.header=v.header;raw.scan_start=Time(nanoseconds=stamp-130_000_000).to_msg()
                raw.sequence=sequence;sequence+=1
                q=r.T@(fixed-p)-np.array([.03,0,-.04])
                raw.points_xyz_m[:3]=q.tolist();raw.hits[0]=1
                scan_pub.publish(raw);last_scan=now
        policy=TwistStamped();policy.header.stamp=Time(nanoseconds=now).to_msg();policy.header.frame_id='base_link'
        policy.twist.linear.x=.4;policy_pub.publish(policy)
        until=time.monotonic()+.02
        while time.monotonic()<until: executor.spin_once(timeout_sec=.002)
    try:
        for _ in range(150):cycle()
        valid=[s for s in snapshots if s.valid]
        assert len(valid)>20
        for snap in valid:
            p,r,_=pose_at(history,ns(snap.header.stamp))
            np.testing.assert_allclose(snap.robot_xy_m,(p+r@np.array([.03,0,-.04])+[.7,-.2,.1])[:2],atol=1e-5)
            np.testing.assert_allclose(snap.scan.points_xy_m[:2],[2.7,.8],atol=1e-5)
            np.testing.assert_allclose(snap.velocity_xy_mps,[.2,.1],atol=1e-5)
        assert any(ns(s.header.stamp)-ns(s.scan.scan_start)>200_000_000 for s in valid)
        assert any(abs(c.twist.linear.x)>.1 for c in commands)
        # Releveling must not move the frozen CBF reference.
        transforms[0].transform.translation.x=10.
        tf.sendTransform(transforms)
        for _ in range(10):cycle()
        np.testing.assert_allclose([s for s in snapshots if s.valid][-1].scan.points_xy_m[:2],[2.7,.8],atol=1e-5)
        # Missing state produces explicit invalid snapshots, then fallback.
        for _ in range(12):cycle(False,False)
        assert any(not s.valid and s.reason=='state_stale' for s in snapshots[-15:])
        client=harness.create_client(Trigger,'/cbf/reset_world')
        assert client.wait_for_service(timeout_sec=1)
        old_session=world.session
        future=client.call_async(Trigger.Request())
        while not future.done():executor.spin_once(timeout_sec=.01)
        assert future.result().success and world.session!=old_session
        assert not world.scans and not world.poses
        for _ in range(25):cycle()
        fresh=[s for s in snapshots if s.valid and s.session_id==world.session]
        assert fresh
        np.testing.assert_allclose(fresh[-1].scan.points_xy_m[:2],[12.,.8],atol=1e-5)
        assert process.poll() is None
        rows=[{v.key:v.value for v in d.status[0].values} for d in diagnostics if d.status]
        assert rows
        healthy=[r for r in rows if r.get('fallback_reason')=='healthy']
        assert healthy
        import json
        report={
            'valid_snapshots':len(valid), 'commands':len(commands),
            'max_solve_ms':max(float(r['solve_time_s'])*1000 for r in rows),
            'max_control_ms':max(float(r['control_elapsed_s'])*1000 for r in rows),
            'healthy_max_constraint_violation':max(float(r['max_constraint_violation']) for r in healthy),
            'deadline_rejections':sum(r['fallback_reason'] in ('timer_late','control_deadline_missed') for r in rows),
        }
        assert report['healthy_max_constraint_violation']<=1e-3
        (tmp_path/'timing.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report))
    finally:
        process.terminate();process.wait(timeout=5);log.close()
        executor.shutdown();world.destroy_node();harness.destroy_node();rclpy.shutdown()


def test_input_ordering_and_clock_reset():
    rclpy.init()
    node=CbfWorldNode()
    try:
        now=node.get_clock().now().nanoseconds
        node.started=now-1_000_000_000
        p=Odometry();p.header.frame_id='camera_init';p.child_frame_id='body'
        p.header.stamp=Time(nanoseconds=now-50_000_000).to_msg();p.pose.pose.orientation.w=1.
        node.pose_callback(p)
        node.pose_callback(p)  # duplicate
        assert len(node.poses)==1
        p.header.stamp=Time(nanoseconds=now-80_000_000).to_msg()
        node.pose_callback(p)
        assert len(node.poses)==1
        p.header.stamp=Time(nanoseconds=now+1_000_000_000).to_msg()
        node.pose_callback(p)
        assert len(node.poses)==1
        p.header.frame_id='unexpected_frame'
        node.pose_callback(p)
        assert len(node.poses)==1
        v=TwistStamped();v.header.frame_id='base_link';v.header.stamp=Time(nanoseconds=now).to_msg()
        v.twist.linear.x=float('nan');node.velocity_callback(v)
        assert not node.velocities
        raw=CbfScan3D();raw.header.frame_id='base_link';raw.header.stamp=Time(nanoseconds=now).to_msg()
        raw.scan_start=Time(nanoseconds=now+1).to_msg();node.scan_callback(raw)
        assert not node.pending
        raw.scan_start=Time(nanoseconds=now-200_000_000).to_msg();node.scan_callback(raw);node.scan_callback(raw)
        assert len(node.pending)==1  # old observations are allowed, duplicates are not
        previous=node.session
        node.last_now=now+1_000_000_000
        node.tick()
        assert node.blocked and node.session!=previous and not node.poses and not node.pending
        response=node.reset_callback(Trigger.Request(),Trigger.Response())
        assert response.success and not node.blocked and node.reference is None
    finally:
        node.destroy_node();rclpy.shutdown()
