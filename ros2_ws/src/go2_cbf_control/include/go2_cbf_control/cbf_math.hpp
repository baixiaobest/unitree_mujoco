#pragma once

#include <algorithm>
#include <cmath>

namespace go2_cbf_control {

inline double zoh_gain(const double step_s, const double tracking_tau_s) {
  return step_s / (1.0 - std::exp(-step_s / tracking_tau_s));
}

inline double static_barrier_offset(
  const double rx, const double ry, const double vx, const double vy,
  const double gamma1, const double gamma2, const double margin_m) {
  const double distance_squared = rx * rx + ry * ry;
  const double velocity_squared = vx * vx + vy * vy;
  return 2.0 * velocity_squared + 2.0 * (gamma1 + gamma2) * (rx * vx + ry * vy) +
         gamma1 * gamma2 * (distance_squared - margin_m * margin_m);
}

inline double approach_zero(const double value, const double maximum_delta) {
  if (value > 0.0) return std::max(0.0, value - maximum_delta);
  return std::min(0.0, value + maximum_delta);
}

}  // namespace go2_cbf_control
