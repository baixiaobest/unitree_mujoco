#pragma once
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <osqp.h>
#include "go2_cbf_control/cbf_math.hpp"

namespace go2_cbf_control {

constexpr int kFrontScanBins = 128;
constexpr int kStaticScanBins = 64;
constexpr int kCandidateBins = kFrontScanBins + kStaticScanBins;
constexpr int kMaxPoints = 64;
constexpr int kVariables = 2 + kMaxPoints;
constexpr int kRows = 2 * kMaxPoints + 2;
constexpr int kBarrierRows = kMaxPoints;
constexpr int kSlackRowStart = kMaxPoints;
constexpr int kAccelRowStart = 2 * kMaxPoints;
// The QP structure never changes, but it is intentionally sparse.  In
// particular, an acceleration only participates in the 64 barrier rows plus
// both rotated bounds, and each slack participates in one barrier and one
// non-negative row.  Do not replace this with a dense 130 x 66 matrix: that
// creates 8,580 stored entries rather than the 260 meaningful entries and
// makes a 5-ms control budget unnecessarily hard to meet.
constexpr int kAccelerationColumnEntries = kMaxPoints + 2;
constexpr int kSlackColumnStart = 2 * kAccelerationColumnEntries;
constexpr int kMatrixNonZeros = kSlackColumnStart + 2 * kMaxPoints;
constexpr double kTolerance = 1.0e-4;

using Clock = std::chrono::steady_clock;

struct Config {
  std::string policy_topic;
  std::string goal_region_topic;
  std::string scan_topic;
  std::string cmd_topic;
  double control_hz{};
  double solver_time_limit_s{};
  double publish_deadline_s{};
  double policy_timeout_s{};
  double velocity_timeout_s{};
  double scan_timeout_s{};
  double max_interpolation_gap_s{};
  double d_margin_m{};
  double d_cbf_active_m{};
  double gamma1{};
  double gamma2{};
  double kp_x{};
  double kp_y{};
  double goal_region_kp{};
  double accel_limit_x{};
  double accel_limit_y{};
  double velocity_limit_x{};
  double velocity_limit_y{};
  double max_yaw_rate_radps{};
  double max_yaw_accel_radps2{};
  bool enable_navigation_slew{};
  double a_nav_mps2{};
  double planar_deadzone_mps{};
  double yaw_deadzone_radps{};
  double tracking_tau_s{};
  int max_lidar_points{};
  double slack_penalty{};
  double max_cbf_slack{};
  double bad_solve_grace_s{};
  double fallback_linear_decel_mps2{};
  double fallback_yaw_decel_radps2{};
};

struct Point {
  double x{};
  double y{};
  double range{};
  bool is_static{};
};

struct SolverResult {
  bool has_candidate{};
  bool solved{};
  bool timed_out_or_iter_limit{};
  bool update_ok{};
  int status{};
  int iterations{};
  double solve_time_s{};
  double u_x{};
  double u_y{};
  double max_slack{};
  double primal_residual{};
  double dual_residual{};
  double max_constraint_violation{};
  std::array<double, kMaxPoints> slack{};
  double update_time_s{};
  std::string status_text{"not_run"};
};

class StaticCbfQp {
 public:
  explicit StaticCbfQp(const Config & config, double eps = 1e-3, int max_iter = 500) : config_(config), eps_(eps), max_iter_(max_iter) {
    if (config_.max_lidar_points != kMaxPoints) {
      throw std::runtime_error("This fixed-sparsity CBF build requires max_lidar_points=64.");
    }
    p_p_.fill(2);
    p_p_[0] = 0;
    p_p_[1] = 1;
    p_p_[2] = 2;
    p_i_[0] = 0;
    p_i_[1] = 1;
    p_x_[0] = 2.0;
    p_x_[1] = 2.0;
    // CSC pattern: accel-x, accel-y, then one two-entry column per slack.
    // Keeping this exact pattern fixed lets OSQP reuse its factorization while
    // every tick updates only numerical values.
    a_p_[0] = 0;
    a_p_[1] = kAccelerationColumnEntries;
    a_p_[2] = kSlackColumnStart;
    for (int row = 0; row < kMaxPoints; ++row) {
      a_i_[row] = row;
      a_i_[kAccelerationColumnEntries + row] = row;
    }
    a_i_[kMaxPoints] = kAccelRowStart;
    a_i_[kMaxPoints + 1] = kAccelRowStart + 1;
    a_i_[kAccelerationColumnEntries + kMaxPoints] = kAccelRowStart;
    a_i_[kAccelerationColumnEntries + kMaxPoints + 1] = kAccelRowStart + 1;
    for (int index = 0; index < kMaxPoints; ++index) {
      const int offset = kSlackColumnStart + 2 * index;
      a_p_[2 + index] = offset;
      a_i_[offset] = index;
      a_i_[offset + 1] = kSlackRowStart + index;
    }
    a_p_[kVariables] = kMatrixNonZeros;
    setup();
  }

  ~StaticCbfQp() {
    if (work_ != nullptr) {
      osqp_cleanup(work_);
    }
  }

  SolverResult solve(
    const double nominal_x, const double nominal_y,
    const std::array<Point, kMaxPoints> & points, const int point_count,
    const double measured_x, const double measured_y, const double effective_margin,
    const double acceleration_lower_x, const double acceleration_lower_y,
    const double acceleration_upper_x, const double acceleration_upper_y, const double yaw = 0.0)
  {
    SolverResult result;
    q_.fill(0.0);
    lower_.fill(-OSQP_INFTY);
    upper_.fill(OSQP_INFTY);
    a_x_.fill(0.0);
    q_[0] = -2.0 * nominal_x;
    q_[1] = -2.0 * nominal_y;

    for (int index = 0; index < point_count; ++index) {
      const auto & point = points[index];
      const double rx = -point.x;
      const double ry = -point.y;
      const double offset = go2_cbf_control::static_barrier_offset(
        rx, ry, measured_x, measured_y, config_.gamma1, config_.gamma2, effective_margin);
      set_a(index, 0, 2.0 * rx);
      set_a(index, 1, 2.0 * ry);
      set_a(index, 2 + index, 1.0);
      lower_[index] = -offset;
      set_a(kSlackRowStart + index, 2 + index, 1.0);
      lower_[kSlackRowStart + index] = 0.0;
      upper_[kSlackRowStart + index] = config_.max_cbf_slack;
      q_[2 + index] = config_.slack_penalty / static_cast<double>(point_count);
    }
    set_a(kAccelRowStart, 0, std::cos(yaw));
    set_a(kAccelRowStart, 1, std::sin(yaw));
    set_a(kAccelRowStart + 1, 0, -std::sin(yaw));
    set_a(kAccelRowStart + 1, 1, std::cos(yaw));
    lower_[kAccelRowStart] = acceleration_lower_x;
    lower_[kAccelRowStart + 1] = acceleration_lower_y;
    upper_[kAccelRowStart] = acceleration_upper_x;
    upper_[kAccelRowStart + 1] = acceleration_upper_y;

    const auto update_start = Clock::now();
    result.update_ok = osqp_update_lin_cost(work_, q_.data()) == 0 &&
      osqp_update_bounds(work_, lower_.data(), upper_.data()) == 0 &&
      osqp_update_A(work_, a_x_.data(), nullptr, static_cast<c_int>(a_x_.size())) == 0;
    result.update_time_s = std::chrono::duration<double>(Clock::now() - update_start).count();
    if (!result.update_ok) {
      result.status_text = "osqp_update_failed";
      return result;
    }
    osqp_solve(work_);
    result.status = work_->info->status_val;
    result.iterations = static_cast<int>(work_->info->iter);
    result.solve_time_s = static_cast<double>(work_->info->solve_time);
    result.primal_residual = static_cast<double>(work_->info->pri_res);
    result.dual_residual = static_cast<double>(work_->info->dua_res);
    result.solved = result.status == OSQP_SOLVED || result.status == OSQP_SOLVED_INACCURATE;
    result.timed_out_or_iter_limit = result.status == OSQP_TIME_LIMIT_REACHED ||
      result.status == OSQP_MAX_ITER_REACHED;
    result.status_text = work_->info->status == nullptr ? "unknown" : work_->info->status;
    if (work_->solution == nullptr || work_->solution->x == nullptr) {
      return result;
    }
    result.u_x = static_cast<double>(work_->solution->x[0]);
    result.u_y = static_cast<double>(work_->solution->x[1]);
    result.has_candidate = std::isfinite(result.u_x) && std::isfinite(result.u_y);
    for (int index = 0; index < point_count; ++index) {
      result.slack[index] = static_cast<double>(work_->solution->x[2 + index]);
      result.max_slack = std::max(result.max_slack, std::max(0.0, result.slack[index]));
    }
    // Evaluate the actual unscaled rows; finite iterates alone are insufficient.
    std::array<double, kRows> ax{};
    for (int col = 0; col < kVariables; ++col) {
      for (int k = a_p_[col]; k < a_p_[col + 1]; ++k) {
        ax[a_i_[k]] += a_x_[k] * work_->solution->x[col];
      }
    }
    for (int row = 0; row < kRows; ++row) {
      if (!std::isfinite(ax[row])) result.max_constraint_violation = std::numeric_limits<double>::infinity();
      result.max_constraint_violation = std::max(result.max_constraint_violation,
        std::max(static_cast<double>(lower_[row]) - ax[row], ax[row] - static_cast<double>(upper_[row])));
    }
    return result;
  }

 private:
  void setup() {
    OSQPData * data = static_cast<OSQPData *>(c_malloc(sizeof(OSQPData)));
    if (data == nullptr) throw std::bad_alloc();
    data->n = kVariables;
    data->m = kRows;
    data->P = csc_matrix(kVariables, kVariables, 2, p_x_.data(), p_i_.data(), p_p_.data());
    data->q = q_.data();
    data->A = csc_matrix(kRows, kVariables, static_cast<c_int>(a_x_.size()), a_x_.data(), a_i_.data(), a_p_.data());
    data->l = lower_.data();
    data->u = upper_.data();
    osqp_set_default_settings(&settings_);
    settings_.verbose = false;
    settings_.warm_start = true;
    // Match Isaac Lab's OSQP setup: warm start plus the solver defaults for
    // adaptive rho and polishing.
    settings_.polish = true;
    settings_.adaptive_rho = true;
    settings_.check_termination = 1;
    settings_.max_iter = max_iter_;
    settings_.eps_abs = eps_;
    settings_.eps_rel = eps_;
    settings_.time_limit = config_.solver_time_limit_s;
    const auto setup_status = osqp_setup(&work_, data, &settings_);
    c_free(data->P);
    c_free(data->A);
    c_free(data);
    if (setup_status != 0) {
      throw std::runtime_error("Could not initialize fixed-sparsity OSQP CBF workspace.");
    }
  }

  void set_a(const int row, const int column, const double value) {
    if (column == 0) {
      a_x_[row >= kAccelRowStart ? kMaxPoints + row - kAccelRowStart : row] = value;
      return;
    }
    if (column == 1) {
      a_x_[kAccelerationColumnEntries + (row >= kAccelRowStart ? kMaxPoints + row - kAccelRowStart : row)] = value;
      return;
    }
    const int slack_index = column - 2;
    a_x_[kSlackColumnStart + 2 * slack_index + (row == slack_index ? 0 : 1)] = value;
  }

  const Config & config_;
  double eps_;
  int max_iter_;
  OSQPWorkspace * work_{};
  OSQPSettings settings_{};
  std::array<c_float, 2> p_x_{};
  std::array<c_int, 2> p_i_{};
  std::array<c_int, kVariables + 1> p_p_{};
  std::array<c_float, kVariables> q_{};
  std::array<c_float, kRows> lower_{};
  std::array<c_float, kRows> upper_{};
  std::array<c_float, kMatrixNonZeros> a_x_{};
  std::array<c_int, kMatrixNonZeros> a_i_{};
  std::array<c_int, kVariables + 1> a_p_{};
};

}  // namespace go2_cbf_control

namespace go2_cbf_control {
inline const char * validate_candidate(const SolverResult & result, int count, double max_slack) {
  if (!result.has_candidate) return "nonfinite_acceleration_candidate";
  if (!std::isfinite(result.max_constraint_violation) || result.max_constraint_violation > 1e-3)
    return "constraint_violation";
  for (int i=0; i<count; ++i) {
    if (!std::isfinite(result.slack[i])) return "nonfinite_slack";
    if (result.slack[i] < -kTolerance) return "negative_slack";
    if (result.slack[i] > max_slack + kTolerance) return "slack_limit_exceeded";
  }
  return nullptr;
}
}
