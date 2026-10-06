# Static observations in cbf_world. Bin IDs refer to capture-time bins.
std_msgs/Header header
builtin_interfaces/Time scan_start
uint32 sequence
string session_id
float64 capture_yaw
float64[256] points_xy_m
uint8[128] hits
float64[128] static_points_xy_m
uint8[64] static_hits
