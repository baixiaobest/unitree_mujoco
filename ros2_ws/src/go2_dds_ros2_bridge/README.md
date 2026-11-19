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
The Go2 RobotModel and animated legs are enabled by default. Pass
`robot_model:=false` to omit the model and joint-state publisher. When enabled,
`/joint_states` carries the 12 motor joints, `/robot_description` carries the
URDF, and `go2_robot_state_publisher` supplies the leg transforms. These are
also available to an RViz instance started separately from the launch.

```bash
# Odometry only
ros2 launch go2_dds_ros2_bridge fast_lio.launch.py run_mode:=hardware rviz:=true

# Build a map
ros2 launch go2_dds_ros2_bridge fast_lio_mapping.launch.py \
  run_mode:=hardware navigation:=false cbf:=false rviz:=true

# Resume mapping from a saved PCD; RViz will open in the map frame
ros2 launch go2_dds_ros2_bridge fast_lio_mapping.launch.py \
  run_mode:=hardware navigation:=false cbf:=false rviz:=true \
  checkpoint_map_path:=/absolute/path/to/map.pcd

# Save the map while mapping is running (run in another terminal)
ros2 service call /map_save std_srvs/srv/Trigger '{}'

# Localize in the saved map
ros2 launch go2_dds_ros2_bridge fast_lio_localization.launch.py \
  run_mode:=hardware navigation:=false cbf:=false rviz:=true
```

The legacy `occupancy_2d` and `occupancy_scan` nodes are disabled by default
in all three launch modes. To use either one, pass `occupancy:=true` or
`occupancy_scan:=true` to the launch command.

The default map is `~/.local/share/go2_fast_lio/maps/map.pcd`. Pass
`map_path:=/absolute/path/to/map.pcd` to the mapping or localization launch
to use another file. For localization, set the initial pose near the robot's
location in the saved map with RViz's **2D Pose Estimate**. ICP waits for this
pose before matching scans; FAST-LIO waits for ICP to initialize. Send another
**2D Pose Estimate** at any time to run ICP again and reset the map alignment.
On an accepted new match, FAST-LIO resets its filter and IMU initialization.
Keep the robot still until localized odometry resumes.

Localization includes mapped obstacles behind the LiDAR origin in CBF by default when `cbf:=true`.
Pass `cbf_map_obstacles:=false` to disable them. RViz's
**CBF Map Floor** and **CBF Map Rear Obstacles** displays show the fitted local
floor and filtered PCD points; **CBF Selected Points** shows the points passed
to the solver. Map points are cleared on a new 2D Pose Estimate and resume
after successful ICP and fresh odometry. The CBF retains 16 of its 64 point
slots for side/rear obstacles when they exist, lending unused slots to the
front. This includes the near-side strip ahead of `base_link` but behind the
forward-mounted LiDAR. Mapped points are considered within 3 m of the LiDAR
by default; change this
in `config/cbf_map_obstacles.yaml` (or pass another file with
`cbf_map_obstacles_config:=/absolute/path/to/config.yaml`) without changing
the live LiDAR CBF range. The PCD cannot detect newly moved obstacles in the
rear blind spot.
The map node also publishes `/cbf/map_filtered_obstacles` (all retained rear
points before 64-bin reduction) and `/cbf/map_rejected_obstacles` (isolated
returns). The localization RViz config shows retained points by default;
enable **CBF Map Rejected Obstacles** to tune `support_radius_m` and
`min_support_points` in `config/cbf_map_obstacles.yaml`.

For resumed mapping, give an RViz **2D Pose Estimate** near the robot after the
checkpoint appears. Mapping and map saving wait for a successful ICP match.
The default output is `map_resumed.pcd` beside the checkpoint; pass
`map_path:=/absolute/path/to/output.pcd` to choose another output. The input
checkpoint is never overwritten. Send another pose estimate to pause mapping
and realign against the accumulated map; a failed match keeps mapping paused.
Use the same `/map_save` service to write the merged map. ICP does not provide
continuous drift correction between pose estimates.

## View a saved map without localization

```bash
ros2 run pcl_ros pcd_to_pointcloud --ros-args \
  -p file_name:=/home/baixiao/.local/share/go2_fast_lio/maps/map.pcd \
  -p tf_frame:=map
```

Open `rviz2` in another terminal. Set **Fixed Frame** to `map`, add a
**PointCloud2** display on `/cloud_pcd`, and use the **Orbit** view for 3D.
