# The observation timestamp is the instant at which the stored scan history was
# reprojected into the current robot frame. scan_stamp is the end time of the
# most recent completed two-cloud scan.
std_msgs/Header header
builtin_interfaces/Time scan_stamp
float32 normalized_scan_age

# Each array is H * W entries, row-major, with H=4 frames ordered newest first
# and W=128 front-FOV bins ordered left to right. Distances are normalized by
# the configured maximum range. A validity value of 0 means the bin is unknown.
float32[512] distances
uint8[512] validity
