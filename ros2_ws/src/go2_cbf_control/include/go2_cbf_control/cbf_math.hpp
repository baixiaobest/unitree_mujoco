#pragma once

#include <algorithm>
#include <cmath>

namespace go2_cbf_control {

enum class FaultResponse {
  kUseCandidate,
  kHoldLastValidCommand,
  kSlewToStop,
};

// A short interruption must not manufacture a braking command: retain the
// last accepted CBF command until the interruption has persisted for the
// configured grace period. A single accepted result resumes normal command
// publication, including if a controlled stop was in progress.
inline FaultResponse fault_response(
  const bool control_tick_healthy, const double consecutive_bad_s, const double grace_s) {
  if (control_tick_healthy) return FaultResponse::kUseCandidate;
  return consecutive_bad_s >= grace_s ? FaultResponse::kSlewToStop :
         FaultResponse::kHoldLastValidCommand;
}

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

// Constrain an angular-velocity target independently of the planar CBF QP.
// The hard rate envelope is applied first, then the output may move from the
// previously published value by at most angular_accel_limit * step_s.  The
// final clamp also makes a changed parameter or an unexpected prior command
// safe immediately.
inline double limited_yaw_command(
  const double target, const double previous, const double max_rate,
  const double max_accel, const double step_s) {
  const double bounded_target = std::clamp(target, -max_rate, max_rate);
  const double maximum_delta = max_accel * std::max(0.0, step_s);
  const double rate_limited = std::clamp(
    bounded_target, previous - maximum_delta, previous + maximum_delta);
  return std::clamp(rate_limited, -max_rate, max_rate);
}

}  // namespace go2_cbf_control
