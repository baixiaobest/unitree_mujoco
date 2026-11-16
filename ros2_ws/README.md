# Go2 ROS 2 workspace

This workspace contains the Go2 DDS bridge, FAST-LIO mapping and localization,
and related ROS 2 packages. See the [bridge command guide](src/go2_dds_ros2_bridge/README.md)
for more launch options and [architecture.html](src/go2_dds_ros2_bridge/architecture.html)
for the system design.

## Build

From this directory:

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-up-to go2_dds_ros2_bridge fast_lio icp_relocalization
source install/setup.bash
```

Source `/opt/ros/humble/setup.bash` and `install/setup.bash` in each new terminal.

## Mapping and localization

The Go2 launches use `/utlidar/time_corrected/cloud` and `/imu/data_fastlio`.
Use `run_mode:=simulation` instead of `hardware` for the simulator.

```bash
# FAST-LIO odometry
ros2 launch go2_dds_ros2_bridge fast_lio.launch.py run_mode:=hardware rviz:=true

# Map the area
ros2 launch go2_dds_ros2_bridge fast_lio_mapping.launch.py \
  run_mode:=hardware navigation:=false cbf:=false rviz:=true

# Save the map while mapping is running, from another terminal
ros2 service call /map_save std_srvs/srv/Trigger '{}'

# Localize in the saved map
ros2 launch go2_dds_ros2_bridge fast_lio_localization.launch.py \
  run_mode:=hardware navigation:=false cbf:=false rviz:=true
```

The default PCD path is `~/.local/share/go2_fast_lio/maps/map.pcd`. Set
`map_path:=/absolute/path/to/map.pcd` on the mapping and localization launches
to use another file. In localization RViz, use **2D Pose Estimate** to mark the
robot's approximate position and heading. ICP waits for this pose before it
initializes FAST-LIO. Send another estimate to relocalize; keep the robot still
until odometry resumes.

## View a saved map without localization

```bash
source /opt/ros/humble/setup.bash
ros2 run pcl_ros pcd_to_pointcloud --ros-args \
  -p file_name:=/home/baixiao/.local/share/go2_fast_lio/maps/map.pcd \
  -p tf_frame:=map
```

In another terminal, run `rviz2`, set **Fixed Frame** to `map`, add a
**PointCloud2** display on `/cloud_pcd`, and use the **Orbit** view for 3D.
