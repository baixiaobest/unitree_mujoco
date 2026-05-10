# ROS2 Workspace

This workspace hosts ROS2 bridge nodes for `simulate_python`.

The first bridge subscribes to the raw DDS odometry published by `GO2HardwareEnvironment`
on `rt/odom` and republishes it as a normal ROS2 `/odom` topic.

## Layout

```text
ros2_ws/
  src/
    go2_dds_ros2_bridge/
```

## Environment Requirements

The bridge process needs access to both:

- ROS2 Python packages (`rclpy`, `nav_msgs`)
- `unitree_sdk2py`

In practice, run it in an environment where both are available. The bridge does not import
`simulate_python`, so it stays decoupled from the control runtime.

## Build

```bash
cd source/unitree_mujoco/ros2_ws
source /opt/ros/humble/setup.bash
rm -rf build install log
colcon build --packages-select go2_dds_ros2_bridge
```

## Run

Hardware:

```bash
cd source/unitree_mujoco/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=0
ros2 run go2_dds_ros2_bridge odom_bridge --run-mode hardware
```

Simulation:

```bash
cd source/unitree_mujoco/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=1
ros2 run go2_dds_ros2_bridge odom_bridge --run-mode simulation
```

The ROS2 domain controls the ROS graph side of the bridge. The raw DDS side uses the bridge
node's `--run-mode`, `--dds-domain-id`, and `--dds-interface` arguments.

## Python Runtime Note

The bridge process must be able to import all of these in the same Python runtime:

- `rclpy`
- `nav_msgs`
- `unitree_sdk2py`
- `cyclonedds`

If `ros2 run` fails with `ModuleNotFoundError: No module named 'cyclonedds'`, that means the
ROS Python environment can see `unitree_sdk2py`, but not the CycloneDDS Python package required
by the generated DDS IDL classes.

On this machine, the Unitree conda environment contains `cyclonedds` for Python 3.12, while ROS
Humble uses Python 3.10. Python 3.10 cannot import Python 3.12 extension modules, so you must
install compatible `cyclonedds` and `unitree_sdk2py` packages into the ROS Python environment,
or use a runtime where both ROS2 and the DDS dependencies are installed for the same Python version.