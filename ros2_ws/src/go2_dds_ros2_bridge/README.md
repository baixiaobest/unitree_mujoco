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

Enable the optional per-scan correspondence topic when launching localization:

```bash
ros2 launch go2_dds_ros2_bridge fast_lio_localization.launch.py \
  run_mode:=hardware navigation:=false cbf:=false map_path:=/absolute/path/to/map.pcd \
  match_debug:=true
```

In another sourced terminal, record the estimator, body IMU, and map alignment:

```bash
ros2 bag record /Odometry /fast_lio/estimator_diagnostics /fast_lio/pose_information /fast_lio/match_debug \
  /imu/data_fastlio_raw /odom /estimated_velocity /tf /tf_static
```

ROS 2 creates the bag folder in that terminal's current working directory.
Start the raw DDS odometry bridge in another sourced terminal before recording
`/odom` and `/estimated_velocity`:

```bash
ros2 run go2_dds_ros2_bridge odom_bridge --run-mode hardware
```

The locomotion controller publishes the underlying DDS odometry and velocity.
Run `pwd` there to locate it. Relative `BAG_DIR` paths passed to the analyzer
are resolved from the analyzer terminal's current working directory.

Start the local Plotly viewer. It opens a browser page and fetches each 3D scan
when selected; it does not generate an HTML file. The **3D scan match** tab is
available whenever the bag contains `/fast_lio/match_debug` and the map TF.
Click **Reference PCD** in that tab to load the same localization map used by
`map_path`, enter its path in **Or enter PCD path** (useful for hidden `~/.local`
folders), or pass `--map-pcd` to preload it. The map is in the `map` frame;
recorded `/tf` places each scan in that frame. Select a scan and click a
matched return, or enter its match number, to query its five map neighbors and
association lines. The selected return and its association appear in the main
3D map.

The analyzer groups time series in Pose, LiDAR matching, Constraints, Body IMU,
and Learned odometry tabs. Each tab loads once and retains its plots and
per-plot smoothing settings. Hover over a time-series plot to show the nearest
sample at the same time on the other visible plots in that tab.
The Pose tab compares per-scan IMU-predicted position and rotation changes
with the existing LiDAR corrections. The position prediction is reconstructed
from `/Odometry` and the logged LiDAR position correction; the rotation plot
uses a pre-update orientation added to `EstimatorDiagnostics`. Older bags
still show the position prediction when both required topics were recorded.
New bags with `/fast_lio/pose_information` show rotation-only and coupled pose
eigenvalues in Constraints. The compact message records the
upper triangle of the 6×6 LiDAR pose information matrix; its columns are world
position xyz followed by IMU-frame rotation xyz. The coupled plot uses a stated
1 m rotation lever arm. Rotation plots include raw and per-match views. With
`MatchDebug` normals, the analyzer also shows raw and normalized translation
constraint plots. The position-only information plot is omitted because it
duplicates the translation constraint plot apart from measurement-noise scaling.
The 3D tab can show the corresponding axes at the robot position; uncheck
**Translation constraint axes** to hide them or change **Axis scale** to adjust
their length. The axes are drawn in both directions because eigenvector sign
has no physical meaning. Rebuild and source the workspace before recording new
bags with the updated message definition.
The 3D tab also shows FAST-LIO and learned-velocity odometry paths from the
selected scan time onward. It aligns the learned path's position and heading
to FAST-LIO at that time. Separate controls toggle each path and the rotation
constraint axes. These paths compare estimates; FAST-LIO is not ground truth.
Older bags still open. Those without pose information omit the new constraint
plots; bags without recorded normals also omit the original translation plots
and 3D axes.

Run the analyzer without a bag argument to open a file-picker window. Use
**Browse** to select the bag folder and optionally preload a reference PCD:

```bash
python3 src/go2_dds_ros2_bridge/scripts/analyze_fast_lio_bag.py
```

The command-line form remains available:

```bash
python3 src/go2_dds_ros2_bridge/scripts/analyze_fast_lio_bag.py BAG_DIR \
  --map-pcd /absolute/path/to/map.pcd
```

The server prints a `http://127.0.0.1:.../` URL. Press Ctrl+C in its terminal to
stop it. Use `--no-open` if you want to open the URL yourself.

The debug topic is absent by default. It records all downsampled returns and
accepted correspondences for every processed scan, so bag size increases when
`match_debug:=true`. `--map-point-limit` controls only how many reference-map
points the browser displays; it does not change the bag data.

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
