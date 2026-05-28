# GO2 DDS ROS2 Bridge: Current Mapping Notes

This file is for other AI agents working inside `source/unitree_mujoco/ros2_ws/src/go2_dds_ros2_bridge`.

## Current Runtime Topology

- Recommended entry point: `launch/fast_lio.launch.py`.
- The main launch now contains:
  - `clock_offset_bridge`
  - `imu_bridge`
  - `tf2_ros/static_transform_publisher` for `base_link -> utlidar_lidar`
  - `tf2_ros/static_transform_publisher` for `body -> base_link`
  - `fast_lio/fastlio_mapping`
  - optional `occupancy_2d`
  - optional `rviz2`
- `odom_bridge_node.py` still exists as a standalone bridge utility, but it is not part of the main FAST-LIO launch.
- KISS-ICP and EKF fusion are no longer part of this package's active launch or build wiring.

## Current LiDAR And IMU Handoff

- `clock_offset_bridge` reads raw DDS `rt/utlidar/cloud` and republishes ROS2 `/utlidar/time_corrected/cloud`.
- `clock_offset_bridge` preserves the incoming cloud frame id. The corrected cloud remains in `utlidar_lidar` unless some downstream node transforms it.
- `imu_bridge` republishes IMU data for FAST-LIO on `/imu/data_fastlio` with frame id `body` in the main launch.
- `clock_offset_bridge` can publish `base_link -> utlidar_lidar` when run standalone, but `fast_lio.launch.py` disables that and uses an explicit static transform publisher instead.

## Current Frame Semantics

- The active static chain in `fast_lio.launch.py` is:
  - `body -> base_link`
  - `base_link -> utlidar_lidar`
- FAST-LIO is configured with `lidar_pose_in_imu_frame()` because the corrected cloud is still expressed in the LiDAR frame.
- Do not swap this to `base_link_pose_in_imu_frame()` unless the incoming cloud is actually transformed first.
- The occupancy mapper consumes `/cloud_registered` in `camera_init` and looks up the current `camera_init -> base_link` and `camera_init -> utlidar_lidar` transforms.

## PointCloud2 And Timebase Constraints

- Preserve `row_step`-aware PointCloud2 decoding. Do not reinterpret the cloud buffer as a flat packed array that ignores `row_step`.
- `clock_offset_bridge` corrects ROS header time for the republished cloud, but it does not rewrite per-point time fields inside `PointCloud2.data`.
- `imu_bridge_node.py` currently stamps outgoing IMU messages from ROS receive time because the Unitree low-state path does not provide a trusted absolute IMU timestamp.

## Occupancy Mapper Behavior

- `occupancy_2d_node.py` subscribes to `/cloud_registered` and publishes `nav_msgs/OccupancyGrid` on `/static_occupancy`.
- The default map is a rolling `15 m x 15 m` window at `0.1 m` resolution.
- The default map frame is `camera_init`.
- The mapper filters points by z in `camera_init` before projection.
- Grid fusion now uses log-odds accumulation plus decay, not binary overwrite.
- Current tunables are exposed through ROS parameters:
  - `hit_log_odds_increment`
  - `miss_log_odds_decrement`
  - `decay_factor`
  - `min_log_odds`
  - `max_log_odds`
- `numba` is optional at runtime. The node falls back to Python implementations if it is unavailable.

## When Editing This Area

- Keep `fast_lio.launch.py` aligned with the actual executable list in `CMakeLists.txt`.
- Do not reintroduce KISS or EKF assumptions into the FAST-LIO launch documentation unless those nodes return to the package.
- Preserve the current LiDAR extrinsic handling: corrected cloud stays in `utlidar_lidar`, and FAST-LIO gets the LiDAR pose in the IMU/body frame.
- Preserve the occupancy mapper's `camera_init`-framed rolling-window logic unless the map semantics are intentionally changing.