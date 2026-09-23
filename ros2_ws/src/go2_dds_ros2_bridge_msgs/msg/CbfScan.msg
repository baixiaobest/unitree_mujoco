# Newest completed, deskewed front scan for the real-time CBF controller.
# The header is in base_link at the end of the completed scan. scan_start
# retains the oldest point time so the controller can apply a conservative
# latency margin.
std_msgs/Header header
builtin_interfaces/Time scan_start
uint32 sequence

# 128 left-to-right front-FOV bins, flattened as [x0, y0, x1, y1, ...].
# hits[i] is one only when the endpoint is a real obstacle return. Free and
# unavailable bins never become CBF obstacle constraints. These stable front
# bin IDs are reserved for future dynamic-obstacle velocity estimation.
float32[256] points_xy_m
uint8[128] hits

# 64 fixed bins across the non-front half-plane, ordered from the left side
# (+90 degrees), via the rear, to the right side (-90 degrees). These are real
# deskewed returns only; they are deliberately static-only and will not
# participate in future lidar velocity estimation.
float32[128] static_points_xy_m
uint8[64] static_hits
