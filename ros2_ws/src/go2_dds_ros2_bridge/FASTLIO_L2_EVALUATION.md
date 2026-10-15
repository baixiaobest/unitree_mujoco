# L2 internal-IMU FAST-LIO evaluation

Build in `ros2_ws` after sourcing ROS 2 Humble:

```bash
colcon build --packages-select fast_lio go2_dds_ros2_bridge
source install/setup.bash
ros2 launch go2_dds_ros2_bridge fastlio_l2.launch.py
```

The standalone launch reads raw Unitree DDS `rt/utlidar/cloud` and `rt/utlidar/imu` on the hardware DDS interface. Use `dds_interface:=...` and `dds_domain_id:=...` to override it. The paired bridge estimates one host-clock offset from L2 IMU arrival times and applies it to both message headers; it leaves each cloud point's relative `time` untouched. It applies the same front-leg mask as the main cloud bridge to make IMU comparisons meaningful. Converted inputs are `/fastlio_l2/input/cloud` (frame `fastlio_l2_lidar`) and `/fastlio_l2/input/imu` (frame `fastlio_l2_imu`).

The estimator settings are in `config/fastlio_l2.yaml`. Online LiDAR-to-L2-IMU extrinsic estimation is enabled. The YAML transform is only its initial guess. Start the robot stationary for IMU initialization, then use varied motion to observe extrinsic convergence. The estimated LiDAR pose in the L2 IMU frame is published as `geometry_msgs/TransformStamped` on `/fastlio_l2/estimated_extrinsic` and is **not** added to TF.

The L2 DDS IMU axes in the recorded Go2 stream do not match the raw cloud axes. In `/tmp/fastlio_l2_debug`, the cloud floor points toward negative LiDAR Z while the stationary IMU acceleration points toward positive IMU Z. The original identity extrinsic made the registered cloud upside down. The current rotation seed uses the mount rotation from `tf_utils.py` plus a 17.3-degree gravity alignment measured from that bag. It is provisional: a replay still moved about 4.2 m from its starting point over 29 seconds, so the pipeline is not yet reliable for position comparison. The bag did not include a simultaneous Go2 body IMU stream, leaving the internal IMU yaw and exact mounting rotation unverified.

Outputs include `/fastlio_l2/odom`, `/fastlio_l2/path`, `/fastlio_l2/cloud_registered`, and `/fastlio_l2/cloud_registered_imu`. TF is independent of the main pipeline:

```text
fastlio_l2_world
└── fastlio_l2_odom
    └── fastlio_l2_imu
        ├── fastlio_l2_go2_imu
        └── fastlio_l2_lidar
```

`fastlio_l2_odom` is FAST-LIO's native world and may be tilted. At startup, the frame adapter waits for 200 stationary L2 IMU samples, then fixes the `fastlio_l2_world → fastlio_l2_odom` rotation so measured gravity is vertical and the initial virtual Go2 IMU heading is zero. It also fixes that edge's translation so `fastlio_l2_go2_imu` starts at `(0, 0, 0)`. Select `fastlio_l2_world` as the RViz fixed frame to see the leveled cloud. The physical Go2 IMU frame can retain a small roll or pitch if the robot is standing on an incline; its pose changes during motion. The two child transforms follow the online extrinsic estimate and the Go2-IMU-to-LiDAR geometry in `tf_utils.py`. The existing `/Odometry`, `/cloud_registered`, `camera_init`, `body`, and downstream consumers are unchanged.

The startup origin is a display convention, not a calibration or agreement between the two world frames. Changes in the estimated extrinsic after startup can shift `fastlio_l2_go2_imu` relative to the initial origin. For comparison, record the odometry, extrinsic, and registered cloud through a stationary–motion–stop sequence.
