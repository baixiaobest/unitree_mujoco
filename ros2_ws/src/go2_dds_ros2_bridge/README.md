# Go2 DDS ROS 2 bridge

Run these commands from `source/unitree_mujoco/ros2_ws`. See
[architecture.html](architecture.html) for the system design.

## Build

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-up-to go2_dds_ros2_bridge fast_lio icp_relocalization
source install/setup.bash
```

Source `/opt/ros/humble/setup.bash` and `install/setup.bash` in each new terminal.

## FAST-LIO

Use `run_mode:=simulation` instead of `hardware` when using the simulator.
Set `rviz:=true` to open the matching RViz view.

```bash
# Odometry only
ros2 launch go2_dds_ros2_bridge fast_lio.launch.py run_mode:=hardware rviz:=true

# Build a map
ros2 launch go2_dds_ros2_bridge fast_lio_mapping.launch.py \
  run_mode:=hardware navigation:=false cbf:=false rviz:=true

# Save the map while mapping is running (run in another terminal)
ros2 service call /map_save std_srvs/srv/Trigger '{}'

# Localize in the saved map
ros2 launch go2_dds_ros2_bridge fast_lio_localization.launch.py \
  run_mode:=hardware navigation:=false cbf:=false rviz:=true
```

The default map is `~/.local/share/go2_fast_lio/maps/map.pcd`. Pass
`map_path:=/absolute/path/to/map.pcd` to the mapping or localization launch
to use another file. For localization, set the initial pose near the robot's
location in the saved map with RViz's **2D Pose Estimate**. ICP waits for this
pose before matching scans; FAST-LIO waits for ICP to initialize. Send another
**2D Pose Estimate** at any time to run ICP again and reset the map alignment.
On an accepted new match, FAST-LIO resets its filter and IMU initialization.
Keep the robot still until localized odometry resumes.

## View a saved map without localization

```bash
ros2 run pcl_ros pcd_to_pointcloud --ros-args \
  -p file_name:=/home/baixiao/.local/share/go2_fast_lio/maps/map.pcd \
  -p tf_frame:=map
```

Open `rviz2` in another terminal. Set **Fixed Frame** to `map`, add a
**PointCloud2** display on `/cloud_pcd`, and use the **Orbit** view for 3D.
