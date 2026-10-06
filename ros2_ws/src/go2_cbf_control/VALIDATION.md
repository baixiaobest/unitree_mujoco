# Migration validation — 2026-10-06

All three modified ROS packages built successfully in the normal `ros2_ws`
workspace. `colcon test` passed for both implementation packages.

| Suite | Passed | Details |
|---|---:|---|
| Existing native controller math | 11 | Tracking, deadzones, yaw governor, fault response |
| World-frame QP | 5 | 200 seeded frame comparisons, 72 analytic projections, asymmetric bounds, slack/infeasibility, candidate rejection |
| World geometry/state alignment | 22 | XYZ transforms, attitude/offset, yaw wrap, time gaps, static held scans, world-frame invariance |
| Existing temporal LiDAR processing | 18 | Regression coverage for the unchanged policy/legacy path |
| Live ROS integration | 2 | Real controller and state node in isolated domain, serialized observations, motion, releveling, reset, ordering, invalid data |
| MuJoCo Go2 trajectories | 6 | 250 prescribed poses each; stationary, approach, retreat, lateral, rotation, combined |

Total: **64 passing tests**. The MuJoCo environment was `unitree` (MuJoCo 3.3.4);
ROS tests used system Python and ROS Humble. MuJoCo tests run separately from
colcon because it is installed in a different Python environment.

A representative isolated ROS run produced 306 valid snapshots and 214 commands:
maximum reported solver time 0.254 ms, maximum control time 0.369 ms, zero sampled
deadline rejections, and zero reported maximum constraint violation on healthy
ticks. These are finite-run diagnostics sampled at the diagnostic publication
rate, not a worst-case execution-time guarantee. Diagnostic decimal formatting
also limits residual/time precision.

Maximum error in the prescribed Go2 transform trajectories was 2.22e-15 m.
QP frame-equivalence fixtures use tighter solver tolerances than deployment to
isolate transform errors; runtime retains the existing solver tolerances.

The synthetic ROS harness exercises the actual native command controller and
world-state node. The MuJoCo scenarios exercise geometric transformations using
actual model poses, not closed-loop gait or collision-avoidance performance.
No robot was driven and no hardware rosbag was available or required for this
migration. Hardware stopping behavior, localization quality and predictor
accuracy are not established by these results.
