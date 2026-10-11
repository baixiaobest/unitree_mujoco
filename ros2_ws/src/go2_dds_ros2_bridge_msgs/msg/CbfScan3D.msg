# Capture-time base_link observations. XYZ retained for full attitude transforms.
std_msgs/Header header
builtin_interfaces/Time scan_start
uint32 sequence
float64[384] points_xyz_m
uint8[128] hits
float64[192] static_points_xyz_m
uint8[64] static_hits
