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

### Inspect FAST-LIO scan matches in a bag

Enable scan correspondences with `match_debug:=true`. Learned velocity fusion
is enabled by default; use `velocity_mode:=imu_lidar` for the original mode.
Set `learned_velocity.stddev` in `config/fast_lio.yaml` to tune its x/y/z
measurement noise (default 0.5 m/s per axis). The `imu_bridge` publishes
`/estimated_velocity` and `/odom` when the controller is running.

```bash
ros2 launch go2_dds_ros2_bridge fast_lio_localization.launch.py \
  run_mode:=hardware navigation:=false cbf:=false map_path:=/absolute/path/to/map.pcd \
  match_debug:=true
```

Record both inputs and diagnostics in another sourced terminal:

```bash
ros2 bag record /Odometry /fast_lio/estimator_diagnostics /fast_lio/pose_information /fast_lio/match_debug \
  /fast_lio/velocity_fusion_debug /utlidar/time_corrected/cloud \
  /imu/data_fastlio /imu/data_fastlio_raw /odom /estimated_velocity /tf /tf_static
```

From the sourced ROS workspace, open the Plotly viewer or rerun a bag with a
different estimator mode:

```bash
python3 src/go2_dds_ros2_bridge/scripts/analyze_fast_lio_bag.py BAG_DIR \
  --map-pcd /absolute/path/to/map.pcd
python3 src/go2_dds_ros2_bridge/scripts/replay_fast_lio_bag.py INPUT_BAG \
  --mode imu_lidar_learned_velocity --output learned_result
python3 src/go2_dds_ros2_bridge/scripts/analyze_fast_lio_bag.py BASELINE_BAG \
  --compare-bag learned_result
```

Omit `BAG_DIR` to choose a bag in the web page; the comparison bag and PCD
can be selected there too. Relative paths use the terminal's current
directory. In **3D scan match**, load the reference PCD,
select a scan, and click a matched return to see its five map neighbors.
Each time-series plot has its own smoothing control. **Compare estimates**
aligns the two paths at their first shared timestamp; neither is ground truth.
The replay result includes its parameters, run manifest, and velocity fusion
acceptance and late-drop counts. Use `--inspect` to check input topics before
replay. Bags without the corrected cloud and processed body IMU cannot be
rerun; the November 25 debug bag is one such example.

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
## Learned motion timestamps and odometry

The velocity-control controller publishes one `rt/learned_motion` DDS sample per policy estimate. It contains the LowState millisecond `tick` captured with the policy observation, a sequence number, body-frame linear and angular velocity, and an angular-source code (`0` learned, `1` raw gyroscope fallback for linear-only policies). The `imu_bridge` maps that tick using the same clock mapping as `/imu/data_fastlio`, then publishes `/estimated_velocity` (`TwistStamped`) and `/odom` (`Odometry`) with matching ROS timestamps. The controller no longer publishes `rt/estimated_velocity` or `rt/odom` directly. Do not also launch `odom_bridge` to publish those ROS topics.

`/odom` starts at zero position and identity orientation and integrates the learned linear velocity and selected angular velocity. Its pose is relative, with no global alignment. Tick repeats do not advance the pose; an input gap over 0.2 seconds skips stale integration. The policy's three-output model uses the gyroscope for angular velocity, while a six-output model uses learned angular velocity.
