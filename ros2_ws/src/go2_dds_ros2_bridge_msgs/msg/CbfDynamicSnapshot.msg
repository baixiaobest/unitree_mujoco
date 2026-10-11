# Predictor output and its entire evaluated state are one atomic observation.
# Front bin i has an obstacle only when reflection_hits[i] is 1.
CbfControlSnapshot base
bool valid
string reason
float64[256] front_points_xy_m
float64[256] front_velocity_xy_mps
uint8[128] reflection_hits
