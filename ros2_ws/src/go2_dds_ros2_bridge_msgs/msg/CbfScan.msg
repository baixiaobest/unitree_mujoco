# Newest completed, deskewed front scan for the real-time CBF controller.
# The header is in base_link at the end of the completed scan. scan_start
# retains the oldest point time so the controller can apply a conservative
# latency margin.
std_msgs/Header header
builtin_interfaces/Time scan_start
uint32 sequence

# 128 left-to-right front-FOV bins, flattened as [x0, y0, x1, y1, ...].
# hits[i] is one only when the endpoint is a real obstacle return. Free and
# unavailable bins never become CBF obstacle constraints.
float32[256] points_xy_m
uint8[128] hits
