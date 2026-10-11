# Atomic measured state and an eligible observation in the same world session.
# header.stamp is evaluation time, NOT publication time.
std_msgs/Header header
bool valid
string reason
string session_id
float64[2] robot_xy_m
float64 robot_yaw
float64[2] velocity_xy_mps
float64 interpolation_gap_s
CbfWorldScan scan
