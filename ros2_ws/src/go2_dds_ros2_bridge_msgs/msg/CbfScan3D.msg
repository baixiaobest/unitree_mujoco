# Capture-time base_link observations. XYZ retained for full attitude transforms.
std_msgs/Header header
builtin_interfaces/Time scan_start
uint32 sequence
float64[384] points_xyz_m
uint8[128] hits
float64[192] static_points_xyz_m
uint8[64] static_hits
# The unreduced sparse front capture. States: 0 unavailable, 1 measured
# no-return, 2 reflection. No-return endpoints are retained for projection.
float64[768] capture_points_xyz_m
uint8[256] ray_states
