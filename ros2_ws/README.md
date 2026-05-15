# ROS2 Workspace

This workspace hosts ROS2 bridge nodes for Unitree DDS topics and the local EKF fusion stack used by the GO2 pipeline.

The bridge package currently provides:

- `odom_bridge`: subscribes to raw DDS odometry on `rt/odom` and republishes it as ROS2 `/odom`
- `clock_offset_bridge`: estimates the GO2 onboard-to-local clock offset from the raw LiDAR cloud header timestamps, publishes the filtered offset, and republishes a corrected ROS2 cloud
- `kiss_odom_node`: runs KISS-ICP on `/utlidar/time_corrected/cloud` and publishes `/kiss/odom`
- `ekf_fusion.launch.py`: composes the full local fusion stack with `robot_localization`

The current fused design is:

- estimator odometry contributes `vx`, `vy`, `yaw_rate`
- KISS odometry contributes `x`, `y`, `yaw`
- the EKF is planar (`two_d_mode = true`)
- the EKF owns the only dynamic `odom -> base_link` transform
- `base_link -> utlidar_lidar` is published as a fixed transform by a static transform publisher in the fusion launch

## Layout

```text
ros2_ws/
  README.md
  src/
    go2_dds_ros2_bridge/
      config/
      launch/
```

## Environment Requirements

The bridge process needs access to both:

- ROS2 Python packages (`rclpy`, `nav_msgs`)
- `unitree_sdk2py`
- `python3-yaml`
- `robot_localization` when using the EKF launch

In practice, run it in an environment where both are available. The bridge does not import
`simulate_python`, so it stays decoupled from the control runtime.

## Build

```bash
cd source/unitree_mujoco/ros2_ws
source /opt/ros/humble/setup.bash
rm -rf build install log
colcon build --packages-select go2_dds_ros2_bridge
```

If you want to use the EKF fusion launch, make sure `robot_localization` is installed in the same ROS environment:

```bash
sudo apt update
sudo apt install ros-humble-robot-localization
```

## Run

### Raw DDS to ROS2 Bridges

Hardware estimator odometry bridge:

```bash
cd source/unitree_mujoco/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=0
ros2 run go2_dds_ros2_bridge odom_bridge --run-mode hardware
```

Hardware clock offset estimator:

```bash
cd source/unitree_mujoco/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=0
ros2 run go2_dds_ros2_bridge clock_offset_bridge \
  --run-mode hardware \
  --output-topic /utlidar/time_corrected/cloud
```

Simulation estimator odometry bridge:

```bash
cd source/unitree_mujoco/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=1
ros2 run go2_dds_ros2_bridge odom_bridge --run-mode simulation
```

The ROS2 domain controls the ROS graph side of the bridge. The raw DDS side uses each bridge
node's `--run-mode`, `--dds-domain-id`, and `--dds-interface` arguments.

The clock offset estimator publishes a low-pass filtered value on `/go2_clock_offset_sec`, where:

```text
local_ros_time ~= go2_onboard_time + go2_clock_offset_sec
```

It uses the raw DDS `rt/utlidar/cloud` topic and reads `PointCloud2.header.stamp` directly, so the
estimated offset comes from the same timestamp stream that downstream ROS tools consume. The same
node republishes the corrected cloud on `/utlidar/time_corrected/cloud`.

### Local EKF Fusion Launch

The recommended runtime entry point is the fusion launch:

```bash
cd source/unitree_mujoco/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=0
ros2 launch go2_dds_ros2_bridge ekf_fusion.launch.py run_mode:=hardware
```

For simulation, switch the ROS domain and launch argument:

```bash
cd source/unitree_mujoco/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=1
ros2 launch go2_dds_ros2_bridge ekf_fusion.launch.py run_mode:=simulation
```

Show launch arguments:

```bash
ros2 launch go2_dds_ros2_bridge ekf_fusion.launch.py --show-args
```

The launch composes:

- `clock_offset_bridge` with its internal TF publication disabled
- `odom_bridge` publishing `/odom`
- `kiss_odom_node` publishing `/kiss/odom`
- `static_transform_publisher` for `base_link -> utlidar_lidar`
- `robot_localization/ekf_node`

The fixed LiDAR extrinsic used by the launch is:

```text
xyz = (0.2929999828338623, 0.0, -0.06000000238418579)
rpy_deg = (192.0, -8.0, -60.0)
```

### EKF and Covariance Configuration

The EKF configuration is stored in:

- `src/go2_dds_ros2_bridge/config/ekf_localization.yaml`

The default diagonal covariance YAML files are:

- `src/go2_dds_ros2_bridge/config/odom_bridge_covariance.yaml`
- `src/go2_dds_ros2_bridge/config/kiss_odom_covariance.yaml`

Each covariance YAML contains variances for these supported state names:

- `x`, `y`, `z`
- `roll`, `pitch`, `yaw`
- `vx`, `vy`, `vz`
- `roll_dt`, `pitch_dt`, `yaw_dt`

Supported aliases in the YAML loader are:

- `x_dt -> vx`
- `y_dt -> vy`
- `z_dt -> vz`
- `vroll -> roll_dt`
- `vpitch -> pitch_dt`
- `vyaw -> yaw_dt`

The current EKF split is:

- `/odom` uses estimator twist only: `vx`, `vy`, `yaw_rate`
- `/kiss/odom` uses KISS pose only: `x`, `y`, `yaw`

You can override the config file paths when launching:

```bash
ros2 launch go2_dds_ros2_bridge ekf_fusion.launch.py \
  run_mode:=hardware \
  ekf_config:=/absolute/path/to/ekf_localization.yaml \
  odom_covariance_config:=/absolute/path/to/odom_bridge_covariance.yaml \
  kiss_covariance_config:=/absolute/path/to/kiss_odom_covariance.yaml
```

If no covariance file is provided when running `odom_bridge` or `kiss_odom_node` directly, each
node uses built-in default diagonal variances.

### TF Ownership

In the current fusion setup:

- the EKF publishes the only dynamic `odom -> base_link` transform
- `base_link -> utlidar_lidar` is fixed and published by `static_transform_publisher`
- `odom_bridge` does not publish TF
- `kiss_odom_node` does not publish TF
- `clock_offset_bridge` can still publish TF when run standalone, but the fusion launch disables it

### Useful Checks

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 topic list | grep -E 'odom|kiss|utlidar|clock_offset'
ros2 topic echo /odometry/filtered --once
ros2 topic echo /odom --once
ros2 topic echo /kiss/odom --once
```

If `robot_localization` is missing, the fusion launch will fail to start `ekf_node`. Check with:

```bash
source /opt/ros/humble/setup.bash
ros2 pkg executables robot_localization
```

## Python Runtime Note

The bridge process must be able to import all of these in the same Python runtime:

- `rclpy`
- `nav_msgs`
- `unitree_sdk2py`
- `cyclonedds`
- `yaml`

If `ros2 run` fails with `ModuleNotFoundError: No module named 'cyclonedds'`, that means the
ROS Python environment can see `unitree_sdk2py`, but not the CycloneDDS Python package required
by the generated DDS IDL classes.

On this machine, the Unitree conda environment contains `cyclonedds` for Python 3.12, while ROS
Humble uses Python 3.10. Python 3.10 cannot import Python 3.12 extension modules, so you must
install compatible `cyclonedds` and `unitree_sdk2py` packages into the ROS Python environment,
or use a runtime where both ROS2 and the DDS dependencies are installed for the same Python version.