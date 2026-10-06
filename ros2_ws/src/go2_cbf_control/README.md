# World-frame CBF

The static CBF optimizes **physical horizontal acceleration in `cbf_world`**.
Policy targets, component limits, and the locomotion command retain the robot's
forward/lateral planar command-axis convention. The QP uses yaw rotation for
those axes; this is not a derivative of rotating body-coordinate velocity.

## Frames and observations

`cbf_world_state` waits for the initial `camera_init_correct <- camera_init`
leveling transform. It freezes that transform for a session, and broadcasts its
inverse as **`camera_init -> cbf_world`**. This avoids giving `camera_init` a
second parent. Later standing/releveling changes do not move the CBF reference.
RViz can continue using `camera_init_correct` as its Fixed Frame.

The pipeline is:

```
completed deskewed XYZ -> /cbf/scan_xyz -> cbf_world_state
  /Odometry + /estimated_velocity -----------^
                    |
                    +-> /cbf/world_scan (capture-time world points)
                    +-> /cbf/snapshot   (atomic evaluated state + eligible scan)
                                                 |
                                            cbf_control
                                                 |
                                              /cmd_vel
```

The old `/cbf/scan` and policy observations remain available. They do not drive
the migrated controller. The new XYZ reduction preserves point height through
the full attitude transform. Rear reduction selects a measured XYZ return
nearest the configured percentile, rather than manufacturing a percentile XY.
Front capture rays retain the existing capture-reduction convention.

World scans contain double-precision XY, hit masks, front/rear bin identities,
capture yaw, reference/oldest-point timestamps, sequence, and session ID.
Full XYZ is transformed before dropping world Z. The `body -> base_link`
translation is included; a nonidentity rotation on that static edge is rejected.
Raw FAST-LIO pose must be `camera_init` / child `body`; velocity must be a
three-component physical linear velocity of the base, expressed in `base_link`.

## Time contract

The snapshot is evaluated at the newest velocity sample for which pose history
covers its attitude. Translation and quaternion attitude are interpolated at
sample times; velocity is transformed with its own attitude. State preparation
runs at 100 Hz and the QP at 50 Hz. There is no pose extrapolation to wall time.

The separate limits in `config/cbf_control.yaml` are:

- Robot state age: 100 ms maximum.
- Pose/velocity interpolation gap: 100 ms maximum, only when interpolation is required.
- Scan age: 500 ms maximum from `scan_start`, not from publication time.

A 200 ms-old scan is valid with a fresh state. Static world points stay fixed,
while relative position and nearest-64 selection are recomputed at every tick.
Future observations cannot be paired with an earlier evaluation state. Missing
TF/state never blocks the control timer; it produces an invalid snapshot.
The controller independently checks frame/session/time consistency.

The same state/scan limits should be configured for both nodes. The old
controller parameter `velocity_timeout_s` is now `state_timeout_s`; its standalone
`velocity_topic` parameter is removed because state arrives atomically.

## Reset and failure behavior

After FAST-LIO restart or a deliberate localization reset, call:

```bash
ros2 service call /cbf/reset_world std_srvs/srv/Trigger '{}'
```

This clears observations and histories, assigns a new session ID, and waits for
fresh samples. A backward ROS-clock jump invalidates the session and requires
this reset. Duplicate/out-of-order samples are ignored. Arbitrary unannounced
SLAM jumps cannot reliably be distinguished from estimated motion; the operator
must reset after an estimator restart. A session change also clears held commands
and reference-governor state in the controller.

QP candidates, including time-limited iterates, must have finite acceleration,
valid slack, and unscaled row violations no larger than `1e-3`. This validates
barrier and rotated acceleration/command-envelope rows. Existing gains, slack
cap, solver defaults, output deadzones, yaw shaping, and ordinary fault
hold/braking behavior are preserved. QP validation does **not** certify the
subsequent deadzone or fallback command.

Diagnostics include state/scan ages, evaluation timestamp, interpolation gap,
session, alignment failure reason, solver residuals and maximum constraint
violation. Candidate/selected clouds and velocity arrows use `cbf_world` at the
snapshot's evaluation timestamp.

## Future dynamic integration

Obstacle velocity is currently zero. A future predictor must associate results
with scan sequence and bin identity, rotate its **absolute** obstacle velocity
from capture-yaw axes to this world frame, and align obstacle position and robot
state to one evaluation timestamp. Predictor integration and obstacle propagation
are not part of this migration.

## Reproduce validation

From `ros2_ws`, with ROS Humble sourced:

```bash
colcon build --packages-select go2_dds_ros2_bridge_msgs go2_cbf_control go2_dds_ros2_bridge --cmake-args -DBUILD_TESTING=ON
source install/setup.bash
colcon test --packages-select go2_cbf_control go2_dds_ros2_bridge --event-handlers console_direct+
colcon test-result --verbose
```

The ROS test uses domain 187 and remaps controller output to `/cbf/test_command`.
It runs real ROS nodes with synthetic state/TF/scan publishers; no robot commands
are sent on the deployment domain. Do not use domain 187 for a robot during tests.

With a Python environment containing MuJoCo, NumPy and pytest, from the repository
root:

```bash
PYTHONPATH=source/unitree_mujoco/ros2_ws/src/go2_dds_ros2_bridge python -m pytest -q -s source/unitree_mujoco/ros2_ws/src/go2_dds_ros2_bridge/test/test_cbf_world_mujoco.py
```

Coverage includes homogeneous-transform oracles, roll/pitch and lever arm,
yaw wrapping, time gaps, frame invariance, stale/future/out-of-order data, session
reset, 200 randomized QP frame comparisons, 72 analytic barrier projections,
asymmetric bounds, slack/infeasibility, and invalid time-limited candidates.
The QP equivalence fixture uses tighter tolerances to isolate coordinate errors;
deployment retains OSQP's existing `1e-3` tolerances and 500 iterations.

MuJoCo tests load the actual Go2 model and evaluate 1,500 prescribed poses over
stationary, approaching, retreating, lateral, turning, and combined trajectories.
These are kinematic transform regressions, **not closed-loop locomotion or
stopping-distance tests**. Hardware replay and stopping validation remain separate.
