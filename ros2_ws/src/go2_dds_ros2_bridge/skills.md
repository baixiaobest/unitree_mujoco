# GO2 DDS ROS2 Bridge: LiDAR Odometry Notes

This file is for other AI agents working inside `source/unitree_mujoco/ros2_ws/src/go2_dds_ros2_bridge`.

## Current LiDAR Odometry Topology

- `clock_offset_bridge` reads raw DDS `rt/utlidar/cloud` and republishes ROS2 `/utlidar/time_corrected/cloud`.
- `clock_offset_bridge` also publishes `base_link -> utlidar_lidar` using:
  - xyz = `(0.2929999828338623, 0.0, -0.06000000238418579)`
  - rpy_deg = `(192.0, -8.0, -60.0)`
- `kiss_odom_node` subscribes to `/utlidar/time_corrected/cloud`, runs KISS-ICP, publishes `/kiss/odom`, and publishes TF `kiss_odom -> kiss_lidar`.
- `kiss_lidar` is a virtual child frame. It is used to avoid conflicting with `utlidar_lidar` in TF.

## Current KISS-ICP Behavior

- PointCloud2 decoding is `row_step`-aware. Do not revert to a flat `width * height` reinterpretation that ignores `row_step`.
- The node extracts `x`, `y`, `z` and one timestamp field from `("t", "timestamp", "timestamps", "time", "time_stamp")`.
- Incoming LiDAR messages carry a per-point `time` field, but it resets every message.
- Because of that reset, accumulated deskew cannot use raw concatenated point times directly.
- The current fix is:
  - accumulate `N` incoming messages (`--accumulate-frames`, current default `10`)
  - shift each message's local point times by its header-stamp offset relative to the oldest message in the batch
  - concatenate shifted point times across the batch
  - normalize to `[0, 1]`
  - pass the normalized timestamps to `KissICP.register_frame(points, timestamps)`
- The published odometry stamp is the newest message header in the accumulated batch.

## Important Operational Findings

- Single-message registration was unstable on this GO2 stream.
- Accumulating multiple incoming clouds before KISS registration made the odometry usable.
- The `time` field spans about one message and resets at every new message, so any accumulated deskew fix must account for header time offsets.
- `clock_offset_bridge` corrects the ROS header stamp of the republished cloud. It does not rewrite the per-point `time` field stored in `PointCloud2.data`.
- `kiss_odom_node` unrotates the fixed LiDAR mounting rotation after registration instead of transforming the incoming cloud. This keeps the KISS branch axis-aligned while leaving the original LiDAR frame untouched.

## Drift and Map History

- KISS-ICP's Python API does not expose a public map-purge method.
- The node supports `--reset-after-registrations` to periodically reinitialize the internal KISS object.
- Reinitialization is done with pose anchoring so the published `/kiss/odom` remains continuous across resets.

## Current TF Semantics

- Hardware / bridge tree:
  - `odom -> base_link -> utlidar_lidar`
- KISS tree:
  - `kiss_odom -> kiss_lidar`
- These trees are intentionally disjoint.
- Do not publish KISS TF on `utlidar_lidar` or `base_link` while the standard bridge TF is active, or TF parent conflicts will return.

## When Editing This Area

- Preserve `row_step`-aware PointCloud2 decoding.
- Preserve accumulated-cloud registration unless the upstream transport changes.
- Preserve header-shifted point-time accumulation if deskew is enabled with `accumulate_frames > 1`.
- If removing debug code, keep the functional timestamp shifting logic.