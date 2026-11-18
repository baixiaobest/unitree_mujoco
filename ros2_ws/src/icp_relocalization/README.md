# ICP & SAC-IA relocalization

1. `icp_node.cpp`: simple icp relocalization
2. `san_ia_gicp`: cpp + ros2 implementation of [Point Cloud Registration Method Based on SAC-IA and NDT Fusion](http://www.jgg09.com/EN/abstract/abstract12346.shtml#).

<p align="center">
    <img src="./result.png" width="50%"/>
</p>

## Go2 initialization

The Go2 localization and checkpoint mapping launches set `scan_stack_size:=5`
on `icp_node`. After an RViz **2D Pose Estimate**, the node collects five
timestamped LiDAR clouds, rotates their points into the last scan's LiDAR
frame using `/imu/data_fastlio`, and runs ICP once on the voxel-filtered
stack. It publishes `/icp_result` only when ICP converges below the configured
fitness threshold. Another pose estimate clears any partial stack.

The node restarts collection if the five scans take more than 0.8 seconds,
accumulated rotation exceeds 3 degrees, acceleration variation exceeds
0.8 m/s² RMS over a 100 ms window, or IMU timestamps do not cover the scans.
The launches read the LiDAR-to-IMU time offset from the FAST-LIO YAML so both
nodes use the same timing. Keep the robot still while initializing; IMU alone
cannot detect steady translation. Outside the Go2 launches, the default
`scan_stack_size` remains one.
