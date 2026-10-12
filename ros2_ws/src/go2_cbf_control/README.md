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

## Switchable dynamic CBF

Set `cbf_mode: static` or `cbf_mode: dynamic` under `cbf_control` in
`config/cbf_control.yaml`, then restart the launch. The current YAML selects dynamic mode.
For dynamic mode, also set `cbf_velocity_predictor.velocity_predictor_jit_path`
to a TorchScript model with `projection_contract.txt` and `num_frames.txt`
metadata. Supported combinations are four frames with `temporal_lidar_v3` or
`temporal_lidar_v4`, and eight frames with `temporal_lidar_v4`. The model is
probed at startup; a missing contract, wrong output shape, or nonfinite output
keeps dynamic prediction idle and the controller on static CBF.
The predictor node is idle in static mode. It never publishes `/cmd_vel`.

The world scan carries the original 256 sparse front rays and states alongside
the legacy 128 front and 64 side/rear CBF bins. The predictor node retains the
model's four or eight newest-first captures, projects them at each snapshot's
evaluation position and yaw, and publishes `/cbf/dynamic_snapshot`. It selects
consecutive capture sequences from the snapshot's embedded scan, independent of
delivery order across ROS topics. Four-frame models keep the existing zero-validity
padding for missing history. Eight-frame models use static CBF until eight
consecutive captures are available after startup, a reset, or a scan gap. The
warmup reason appears as `insufficient_predictor_history` in `/cbf/status`.
Its 128 output velocities are
**absolute obstacle velocities in evaluation-yaw axes**. Only newest-frame
reflection winners become CBF points. Each winner's velocity is rotated into
`cbf_world`; its measured world position is advanced by velocity times the
capture-end-to-evaluation interval. Side/rear returns retain zero velocity.
The ROS propagation differs from the current Isaac Lab PLAY dynamic CBF, which
uses measured held-scan positions.

The C++ QP uses robot minus obstacle velocity in the barrier terms. Invalid or
stale dynamic data triggers the existing static CBF, with its reason shown in
`/cbf/status`. If the static snapshot also fails, the existing hold-then-stop
path applies. A session reset discards predictor history. Predictor inference
runs in its own latest-snapshot worker, outside the 50 Hz control timer.
In RViz, add `/cbf/debug_markers` as a MarkerArray and set the fixed frame to
`cbf_world`. In dynamic mode, cyan spheres show every reflected front bin,
including bins outside the active CBF radius. Magenta arrows start at the
matching sphere and show its predicted world velocity at a scale of 1 m per
1 m/s. The `cbf_front_bin_position` and `cbf_front_bin_velocity` marker IDs
are the 0–127 bin indices. A sphere without an arrow has effectively zero
predicted velocity. `/cbf/candidate_points` and `/cbf/selected_points` show
the smaller sets admitted to the CBF radius and QP; all use `cbf_world`.
The per-bin markers expire when dynamic mode falls back to static.
If RViz shows no markers, confirm that the `cbf_control` node is running and
that `/cbf/debug_markers` has a subscriber. Set the RViz MarkerArray display's
Reliability Policy to Best Effort. The markers use `cbf_world`; RViz can use
`cbf_world` or a connected fixed frame. `/cbf/status` reports `dynamic_active` and
`dynamic_reason`; static mode shows command arrows but no per-bin spheres or
obstacle-velocity arrows.

The configured `lidar_velocity_predictor_4_frames_jit.pt` declares
`temporal_lidar_v3` and is accepted as a four-frame model. Validate its
prediction quality on deployment scans before relying on dynamic constraints.
To use the existing eight-frame artifact, change only
`cbf_velocity_predictor.velocity_predictor_jit_path` to
`lidar_velocity_predictor_8_frames_jit.pt` and restart the launch. Restore the
four-frame path and restart to roll back. Both capture histories use the roughly
130 ms completed-scan cadence; the policy observation remains four frames.
Before selecting eight frames on the robot, replay recorded scans to compare ROS
projection with the training projector and check finite predictions, dynamic
activation after warmup, predictor latency, and CBF freshness fallbacks. The node
logs its selected path, contract, frame count, startup probe time, and periodic
inference mean/maximum latency. It warns when total prediction processing exceeds
50 ms. A local shape check does not establish model
accuracy or timing on the deployment computer.

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
