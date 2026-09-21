#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <diagnostic_msgs/msg/key_value.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <rclcpp/create_timer.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <std_msgs/msg/bool.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <go2_dds_ros2_bridge_msgs/msg/cbf_scan.hpp>
#include <osqp.h>

#include "go2_cbf_control/cbf_math.hpp"

namespace {

constexpr int kScanBins = 128;
constexpr int kMaxPoints = 64;
constexpr int kVariables = 2 + kMaxPoints;
constexpr int kRows = 2 * kMaxPoints + 2;
constexpr int kBarrierRows = kMaxPoints;
constexpr int kSlackRowStart = kMaxPoints;
constexpr int kAccelRowStart = 2 * kMaxPoints;
// The QP structure never changes, but it is intentionally sparse.  In
// particular, an acceleration only participates in the 64 barrier rows plus
// its own bound, and each slack participates in one barrier and one
// non-negative row.  Do not replace this with a dense 130 x 66 matrix: that
// creates 8,580 stored entries rather than the 258 meaningful entries and
// makes a 5-ms control budget unnecessarily hard to meet.
constexpr int kAccelerationColumnEntries = kMaxPoints + 1;
constexpr int kSlackColumnStart = 2 * kAccelerationColumnEntries;
constexpr int kMatrixNonZeros = kSlackColumnStart + 2 * kMaxPoints;
constexpr double kTolerance = 1.0e-4;

using Clock = std::chrono::steady_clock;

struct Config {
  std::string policy_topic;
  std::string goal_region_topic;
  std::string velocity_topic;
  std::string scan_topic;
  std::string cmd_topic;
  double control_hz{};
  double solver_time_limit_s{};
  double publish_deadline_s{};
  double policy_timeout_s{};
  double velocity_timeout_s{};
  double scan_timeout_s{};
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
  double tracking_tau_s{};
  int max_lidar_points{};
  double slack_penalty{};
  double max_cbf_slack{};
  double fallback_linear_decel_mps2{};
  double fallback_yaw_decel_radps2{};
  int recovery_valid_cycles{};
};

struct Point {
  double x{};
  double y{};
  double range{};
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
  std::array<double, kMaxPoints> slack{};
  double update_time_s{};
  const char * status_text{"not_run"};
};

class StaticCbfQp {
 public:
  explicit StaticCbfQp(const Config & config) : config_(config) {
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
    a_i_[2 * kMaxPoints + 1] = kAccelRowStart + 1;
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
    const double acceleration_upper_x, const double acceleration_upper_y)
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
    set_a(kAccelRowStart, 0, 1.0);
    set_a(kAccelRowStart + 1, 1, 1.0);
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
    result.status_text = work_->info->status;
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
    settings_.max_iter = 500;
    settings_.eps_abs = 1.0e-3;
    settings_.eps_rel = 1.0e-3;
    settings_.time_limit = config_.solver_time_limit_s;
    if (osqp_setup(&work_, data, &settings_) != 0) {
      throw std::runtime_error("Could not initialize fixed-sparsity OSQP CBF workspace.");
    }
  }

  void set_a(const int row, const int column, const double value) {
    if (column == 0) {
      a_x_[row == kAccelRowStart ? kMaxPoints : row] = value;
      return;
    }
    if (column == 1) {
      a_x_[kAccelerationColumnEntries + (row == kAccelRowStart + 1 ? kMaxPoints : row)] = value;
      return;
    }
    const int slack_index = column - 2;
    a_x_[kSlackColumnStart + 2 * slack_index + (row == slack_index ? 0 : 1)] = value;
  }

  const Config & config_;
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

class CbfControlNode final : public rclcpp::Node {
 public:
  CbfControlNode() : Node("cbf_control"), config_(read_config()), qp_(config_) {
    if (config_.fallback_linear_decel_mps2 <= 0.0 || config_.fallback_yaw_decel_radps2 <= 0.0) {
      throw std::runtime_error(
        "CBF requires positive fallback_linear_decel_mps2 and "
        "fallback_yaw_decel_radps2 parameters.");
    }
    io_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    control_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    debug_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    rclcpp::SubscriptionOptions input_options;
    input_options.callback_group = io_group_;
    const auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();
    policy_sub_ = create_subscription<geometry_msgs::msg::TwistStamped>(
      config_.policy_topic, qos, [this](geometry_msgs::msg::TwistStamped::SharedPtr message) {
        std::lock_guard<std::mutex> lock(policy_mutex_); policy_ = std::move(message);
      }, input_options);
    goal_region_sub_ = create_subscription<std_msgs::msg::Bool>(
      config_.goal_region_topic, qos, [this](std_msgs::msg::Bool::SharedPtr message) {
        std::lock_guard<std::mutex> lock(goal_region_mutex_);
        in_goal_region_ = message->data;
      }, input_options);
    velocity_sub_ = create_subscription<geometry_msgs::msg::TwistStamped>(
      config_.velocity_topic, qos, [this](geometry_msgs::msg::TwistStamped::SharedPtr message) {
        std::lock_guard<std::mutex> lock(velocity_mutex_); velocity_ = std::move(message);
      }, input_options);
    scan_sub_ = create_subscription<go2_dds_ros2_bridge_msgs::msg::CbfScan>(
      config_.scan_topic, qos, [this](go2_dds_ros2_bridge_msgs::msg::CbfScan::SharedPtr message) {
        std::lock_guard<std::mutex> lock(scan_mutex_); scan_ = std::move(message);
      }, input_options);
    command_pub_ = create_publisher<geometry_msgs::msg::TwistStamped>(config_.cmd_topic, rclcpp::QoS(1));
    status_pub_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>("/cbf/status", rclcpp::QoS(10));
    candidate_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("/cbf/candidate_points", qos);
    selected_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("/cbf/selected_points", qos);
    marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("/cbf/debug_markers", qos);
    const auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::duration<double>(1.0 / config_.control_hz));
    control_timer_ = rclcpp::create_wall_timer(
      period, std::bind(&CbfControlNode::control_step, this), control_group_, get_node_base_interface().get(),
      get_node_timers_interface().get());
    debug_timer_ = rclcpp::create_wall_timer(
      std::chrono::milliseconds(100), std::bind(&CbfControlNode::publish_debug, this), debug_group_,
      get_node_base_interface().get(), get_node_timers_interface().get());
    last_control_ = Clock::now();
    RCLCPP_INFO(
      get_logger(), "Static CBF active: %.1f Hz, OSQP limit=%.3f ms, publish deadline=%.3f ms.",
      config_.control_hz, config_.solver_time_limit_s * 1e3,
      config_.publish_deadline_s * 1e3);
  }

 private:
  Config read_config() {
    Config config;
    config.policy_topic = declare_parameter<std::string>("policy_topic", "/policy_vel");
    config.goal_region_topic = declare_parameter<std::string>(
      "goal_region_topic", "/navigation/in_goal_region");
    config.velocity_topic = declare_parameter<std::string>("velocity_topic", "/estimated_velocity");
    config.scan_topic = declare_parameter<std::string>("scan_topic", "/cbf/scan");
    config.cmd_topic = declare_parameter<std::string>("cmd_topic", "/cmd_vel");
    config.control_hz = declare_parameter<double>("control_hz", 50.0);
    config.solver_time_limit_s = declare_parameter<double>("solver_time_limit_s", 0.005);
    config.publish_deadline_s = declare_parameter<double>("publish_deadline_s", 0.010);
    config.policy_timeout_s = declare_parameter<double>("policy_timeout_s", 0.25);
    config.velocity_timeout_s = declare_parameter<double>("velocity_timeout_s", 0.10);
    config.scan_timeout_s = declare_parameter<double>("scan_timeout_s", 0.25);
    config.d_margin_m = declare_parameter<double>("d_margin_m", 0.70);
    config.d_cbf_active_m = declare_parameter<double>("d_cbf_active_m", 5.0);
    config.gamma1 = declare_parameter<double>("gamma1", 2.0);
    config.gamma2 = declare_parameter<double>("gamma2", 2.0);
    config.kp_x = declare_parameter<double>("kp_x", 8.0);
    config.kp_y = declare_parameter<double>("kp_y", 8.0);
    config.goal_region_kp = declare_parameter<double>("goal_region_kp", 2.0);
    config.accel_limit_x = declare_parameter<double>("accel_limit_x", 5.0);
    config.accel_limit_y = declare_parameter<double>("accel_limit_y", 5.0);
    config.velocity_limit_x = declare_parameter<double>("velocity_limit_x", 1.5);
    config.velocity_limit_y = declare_parameter<double>("velocity_limit_y", 1.5);
    config.tracking_tau_s = declare_parameter<double>("tracking_tau_s", 0.30);
    config.max_lidar_points = declare_parameter<int>("max_lidar_points", 64);
    config.slack_penalty = declare_parameter<double>("slack_penalty", 1000.0);
    config.max_cbf_slack = declare_parameter<double>("max_cbf_slack", 0.05);
    config.fallback_linear_decel_mps2 = declare_parameter<double>("fallback_linear_decel_mps2", 0.5);
    config.fallback_yaw_decel_radps2 = declare_parameter<double>("fallback_yaw_decel_radps2", 1.0);
    config.recovery_valid_cycles = declare_parameter<int>("recovery_valid_cycles", 3);
    if (config.control_hz <= 0.0 || config.solver_time_limit_s <= 0.0 || config.publish_deadline_s <= config.solver_time_limit_s ||
        config.tracking_tau_s <= 0.0 || config.d_margin_m <= 0.0 || config.d_cbf_active_m < config.d_margin_m ||
        config.gamma1 <= 0.0 || config.gamma2 <= 0.0 || config.kp_x <= 0.0 || config.kp_y <= 0.0 ||
        config.goal_region_kp <= 0.0 ||
        config.accel_limit_x <= 0.0 || config.accel_limit_y <= 0.0 || config.velocity_limit_x <= 0.0 ||
        config.velocity_limit_y <= 0.0 || config.slack_penalty <= 0.0 ||
        config.max_cbf_slack < 0.0 ||
        config.recovery_valid_cycles < 1 || config.goal_region_topic.empty()) {
      throw std::runtime_error("Invalid static CBF timing or physical parameters.");
    }
    return config;
  }

  bool try_copy_policy(std::shared_ptr<geometry_msgs::msg::TwistStamped> & destination) {
    std::unique_lock<std::mutex> lock(policy_mutex_, std::try_to_lock);
    if (!lock.owns_lock() || !policy_) return false;
    destination = policy_;
    return true;
  }

  bool try_copy_velocity(std::shared_ptr<geometry_msgs::msg::TwistStamped> & destination) {
    std::unique_lock<std::mutex> lock(velocity_mutex_, std::try_to_lock);
    if (!lock.owns_lock() || !velocity_) return false;
    destination = velocity_;
    return true;
  }

  bool try_copy_scan(std::shared_ptr<go2_dds_ros2_bridge_msgs::msg::CbfScan> & destination) {
    std::unique_lock<std::mutex> lock(scan_mutex_, std::try_to_lock);
    if (!lock.owns_lock() || !scan_) return false;
    destination = scan_;
    return true;
  }

  std::optional<bool> try_copy_goal_region() {
    std::unique_lock<std::mutex> lock(goal_region_mutex_, std::try_to_lock);
    if (!lock.owns_lock()) return std::nullopt;
    return in_goal_region_;
  }

  double ros_age_s(const builtin_interfaces::msg::Time & stamp) const {
    const auto stamp_ns = rclcpp::Time(stamp).nanoseconds();
    if (stamp_ns <= 0) return std::numeric_limits<double>::infinity();
    return std::max(0.0, (now().nanoseconds() - stamp_ns) * 1.0e-9);
  }

  bool select_points(
    const go2_dds_ros2_bridge_msgs::msg::CbfScan & scan,
    std::array<Point, kMaxPoints> & selected, int & selected_count,
    std::array<Point, kScanBins> & candidates, int & candidate_count) const
  {
    candidate_count = 0;
    for (int index = 0; index < kScanBins; ++index) {
      if (scan.hits[index] > 1) return false;
      if (scan.hits[index] == 0) continue;
      const double x = scan.points_xy_m[2 * index];
      const double y = scan.points_xy_m[2 * index + 1];
      const double range = std::hypot(x, y);
      if (!std::isfinite(x) || !std::isfinite(y) || range <= 1.0e-4 || range > config_.d_cbf_active_m) continue;
      candidates[candidate_count++] = {x, y, range};
    }
    std::sort(candidates.begin(), candidates.begin() + candidate_count, [](const Point & left, const Point & right) {
      return left.range < right.range;
    });
    selected_count = std::min(candidate_count, kMaxPoints);
    for (int index = 0; index < selected_count; ++index) selected[index] = candidates[index];
    return true;
  }

  // Both solved and time-limited OSQP iterates use this deliberately small
  // accept/reject gate: finite values and the configured slack cap.  The
  // command envelope is clamped when the command is formed below.
  const char * normal_solution_validation_failure(const SolverResult & result, const int count) const
  {
    if (!result.has_candidate) return "nonfinite_acceleration_candidate";
    for (int index = 0; index < count; ++index) {
      const double slack = result.slack[index];
      if (!std::isfinite(slack)) return "nonfinite_slack";
      if (slack < -kTolerance) return "negative_slack";
      if (slack > config_.max_cbf_slack + kTolerance) return "slack_limit_exceeded";
    }
    return nullptr;
  }

  void control_step() {
    const auto start = Clock::now();
    const double expected_step_s = 1.0 / config_.control_hz;
    const double step_s = std::max(0.0, std::chrono::duration<double>(start - last_control_).count());
    const double timer_lateness_s = std::max(0.0, step_s - expected_step_s);
    last_control_ = start;
    std::shared_ptr<geometry_msgs::msg::TwistStamped> policy;
    std::shared_ptr<geometry_msgs::msg::TwistStamped> velocity;
    std::shared_ptr<go2_dds_ros2_bridge_msgs::msg::CbfScan> scan;
    const bool in_goal_region = try_copy_goal_region().value_or(false);
    const bool inputs_available = try_copy_policy(policy) && try_copy_velocity(velocity) && try_copy_scan(scan);
    const double policy_age = policy ? ros_age_s(policy->header.stamp) : std::numeric_limits<double>::infinity();
    const double velocity_age = velocity ? ros_age_s(velocity->header.stamp) : std::numeric_limits<double>::infinity();
    const double scan_age = scan ? ros_age_s(scan->scan_start) : std::numeric_limits<double>::infinity();
    const bool valid_headers = policy && velocity && scan && policy->header.frame_id == "base_link" &&
      velocity->header.frame_id == "base_link" && scan->header.frame_id == "base_link" &&
      rclcpp::Time(scan->header.stamp).nanoseconds() >= rclcpp::Time(scan->scan_start).nanoseconds();
    bool healthy = inputs_available && valid_headers && policy_age <= config_.policy_timeout_s &&
      velocity_age <= config_.velocity_timeout_s && scan_age <= config_.scan_timeout_s && timer_lateness_s <= config_.publish_deadline_s;
    const char * fallback_reason = "healthy";
    if (!inputs_available) fallback_reason = "input_snapshot_unavailable";
    else if (!valid_headers) fallback_reason = "invalid_frame_or_scan_timestamps";
    else if (policy_age > config_.policy_timeout_s) fallback_reason = "policy_stale";
    else if (velocity_age > config_.velocity_timeout_s) fallback_reason = "velocity_stale";
    else if (scan_age > config_.scan_timeout_s) fallback_reason = "scan_stale";
    else if (timer_lateness_s > config_.publish_deadline_s) fallback_reason = "timer_late";
    std::array<Point, kMaxPoints> selected{};
    int selected_count = 0;
    std::array<Point, kScanBins> candidates{};
    int candidate_count = 0;
    SolverResult result;
    double nominal_x = 0.0, nominal_y = 0.0, margin = config_.d_margin_m;
    double measured_x = 0.0, measured_y = 0.0, command_x = 0.0, command_y = 0.0, command_wz = 0.0;
    if (healthy) {
      measured_x = velocity->twist.linear.x;
      measured_y = velocity->twist.linear.y;
      if (!std::isfinite(measured_x) || !std::isfinite(measured_y) ||
          !std::isfinite(policy->twist.linear.x) || !std::isfinite(policy->twist.linear.y) ||
          !std::isfinite(policy->twist.angular.z)) {
        healthy = false;
        fallback_reason = "nonfinite_input";
      }
    }
    if (healthy) {
      healthy = select_points(*scan, selected, selected_count, candidates, candidate_count);
      if (!healthy) fallback_reason = "malformed_scan";
    }
    if (healthy) {
      const double zoh = go2_cbf_control::zoh_gain(step_s, config_.tracking_tau_s);
      const double kp_x = in_goal_region ? config_.goal_region_kp : config_.kp_x;
      const double kp_y = in_goal_region ? config_.goal_region_kp : config_.kp_y;
      nominal_x = std::clamp(kp_x * (policy->twist.linear.x - measured_x), -config_.accel_limit_x, config_.accel_limit_x);
      nominal_y = std::clamp(kp_y * (policy->twist.linear.y - measured_y), -config_.accel_limit_y, config_.accel_limit_y);
      const double lower_x = std::max(-config_.accel_limit_x, (-config_.velocity_limit_x - measured_x) / zoh);
      const double lower_y = std::max(-config_.accel_limit_y, (-config_.velocity_limit_y - measured_y) / zoh);
      const double upper_x = std::min(config_.accel_limit_x, (config_.velocity_limit_x - measured_x) / zoh);
      const double upper_y = std::min(config_.accel_limit_y, (config_.velocity_limit_y - measured_y) / zoh);
      if (lower_x > upper_x || lower_y > upper_y) {
        healthy = false;
        fallback_reason = "empty_command_envelope";
      } else if (selected_count == 0) {
        command_x = std::clamp(measured_x + zoh * nominal_x, -config_.velocity_limit_x, config_.velocity_limit_x);
        command_y = std::clamp(measured_y + zoh * nominal_y, -config_.velocity_limit_y, config_.velocity_limit_y);
        result.solved = true;
        result.update_ok = true;
        result.status_text = "no_active_obstacles";
      } else {
        result = qp_.solve(nominal_x, nominal_y, selected, selected_count, measured_x, measured_y, margin,
          lower_x, lower_y, upper_x, upper_y);
        const char * validation_failure = normal_solution_validation_failure(result, selected_count);
        bool candidate_valid = validation_failure == nullptr;
        healthy = result.update_ok && candidate_valid && (result.solved || result.timed_out_or_iter_limit);
        if (!healthy) {
          if (!result.update_ok) fallback_reason = "osqp_update_failed";
          else if (result.timed_out_or_iter_limit) fallback_reason = "osqp_timeout_candidate_invalid";
          else if (!result.solved) fallback_reason = "osqp_not_solved";
          else fallback_reason = validation_failure == nullptr ? "osqp_solution_invalid" : validation_failure;
        }
        if (healthy) {
          command_x = std::clamp(measured_x + zoh * result.u_x, -config_.velocity_limit_x, config_.velocity_limit_x);
          command_y = std::clamp(measured_y + zoh * result.u_y, -config_.velocity_limit_y, config_.velocity_limit_y);
        }
      }
      command_wz = policy->twist.angular.z;
    }
    const double elapsed_s = std::chrono::duration<double>(Clock::now() - start).count();
    if (elapsed_s > config_.publish_deadline_s) {
      healthy = false;
      fallback_reason = "control_deadline_missed";
    }
    if (result.timed_out_or_iter_limit) ++timeout_count_;
    if (!healthy) {
      if (!fallback_active_) ++fallback_transitions_;
      fallback_active_ = true;
      healthy_cycles_ = 0;
      command_x = go2_cbf_control::approach_zero(last_command_x_, config_.fallback_linear_decel_mps2 * step_s);
      command_y = go2_cbf_control::approach_zero(last_command_y_, config_.fallback_linear_decel_mps2 * step_s);
      command_wz = go2_cbf_control::approach_zero(last_command_wz_, config_.fallback_yaw_decel_radps2 * step_s);
    } else if (fallback_active_) {
      ++healthy_cycles_;
      if (healthy_cycles_ < config_.recovery_valid_cycles) {
        fallback_reason = "recovery_waiting_for_healthy_cycles";
        command_x = go2_cbf_control::approach_zero(last_command_x_, config_.fallback_linear_decel_mps2 * step_s);
        command_y = go2_cbf_control::approach_zero(last_command_y_, config_.fallback_linear_decel_mps2 * step_s);
        command_wz = go2_cbf_control::approach_zero(last_command_wz_, config_.fallback_yaw_decel_radps2 * step_s);
      } else {
        fallback_active_ = false;
        ++fallback_transitions_;
      }
    }
    const auto publish_start = Clock::now();
    publish_command(command_x, command_y, command_wz);
    const double release_to_publish_s = std::chrono::duration<double>(Clock::now() - publish_start).count();
    last_command_x_ = command_x;
    last_command_y_ = command_y;
    last_command_wz_ = command_wz;
    update_debug(policy_age, velocity_age, scan_age, margin, in_goal_region, policy ? policy->twist.linear.x : 0.0,
      policy ? policy->twist.linear.y : 0.0, nominal_x, nominal_y, measured_x, measured_y,
      command_x, command_y, command_wz, elapsed_s, timer_lateness_s, release_to_publish_s, result,
      candidates, candidate_count, selected, selected_count, timeout_count_, fallback_transitions_, fallback_reason);
  }

  void publish_command(const double x, const double y, const double wz) {
    geometry_msgs::msg::TwistStamped command;
    command.header.stamp = now();
    command.header.frame_id = "base_link";
    command.twist.linear.x = x;
    command.twist.linear.y = y;
    command.twist.angular.z = wz;
    command_pub_->publish(command);
  }

  void update_debug(
    double policy_age, double velocity_age, double scan_age, double margin, bool in_goal_region,
    double policy_x, double policy_y,
    double nominal_x, double nominal_y,
    double measured_x, double measured_y, double command_x, double command_y, double command_wz, double elapsed_s,
    double timer_lateness_s, double release_to_publish_s, const SolverResult & result,
    const std::array<Point, kScanBins> & candidates, int candidate_count,
    const std::array<Point, kMaxPoints> & selected, int selected_count,
    uint64_t timeout_count, uint64_t fallback_transitions, const char * fallback_reason)
  {
    std::unique_lock<std::mutex> lock(debug_mutex_, std::try_to_lock);
    if (!lock.owns_lock()) return;
    debug_.policy_age = policy_age; debug_.velocity_age = velocity_age; debug_.scan_age = scan_age; debug_.margin = margin;
    debug_.in_goal_region = in_goal_region;
    debug_.policy_x = policy_x; debug_.policy_y = policy_y;
    debug_.nominal_x = nominal_x; debug_.nominal_y = nominal_y; debug_.measured_x = measured_x; debug_.measured_y = measured_y;
    debug_.command_x = command_x; debug_.command_y = command_y; debug_.command_wz = command_wz; debug_.elapsed_s = elapsed_s;
    debug_.timer_lateness_s = timer_lateness_s; debug_.release_to_publish_s = release_to_publish_s;
    debug_.timeout_count = timeout_count; debug_.fallback_transitions = fallback_transitions;
    debug_.result = result; debug_.fallback = fallback_active_; debug_.fallback_reason = fallback_reason; debug_.candidates = candidates;
    debug_.candidate_count = candidate_count; debug_.selected = selected; debug_.selected_count = selected_count;
  }

  sensor_msgs::msg::PointCloud2 make_cloud(const Point * points, const std::size_t point_count, const uint32_t rgba) const {
    sensor_msgs::msg::PointCloud2 cloud;
    cloud.header.stamp = now(); cloud.header.frame_id = "base_link";
    cloud.height = 1; cloud.width = static_cast<uint32_t>(point_count); cloud.is_dense = true; cloud.is_bigendian = false;
    cloud.fields.resize(4);
    const std::array<std::string, 4> names{"x", "y", "z", "rgba"};
    const std::array<uint8_t, 4> types{sensor_msgs::msg::PointField::FLOAT32, sensor_msgs::msg::PointField::FLOAT32,
      sensor_msgs::msg::PointField::FLOAT32, sensor_msgs::msg::PointField::UINT32};
    for (std::size_t index = 0; index < cloud.fields.size(); ++index) {
      cloud.fields[index].name = names[index]; cloud.fields[index].offset = static_cast<uint32_t>(index * 4);
      cloud.fields[index].datatype = types[index]; cloud.fields[index].count = 1;
    }
    cloud.point_step = 16; cloud.row_step = cloud.point_step * cloud.width; cloud.data.resize(cloud.row_step);
    for (std::size_t index = 0; index < point_count; ++index) {
      const float x = static_cast<float>(points[index].x), y = static_cast<float>(points[index].y), z = 0.0F;
      auto * destination = cloud.data.data() + index * cloud.point_step;
      std::memcpy(destination, &x, sizeof(x)); std::memcpy(destination + 4, &y, sizeof(y));
      std::memcpy(destination + 8, &z, sizeof(z)); std::memcpy(destination + 12, &rgba, sizeof(rgba));
    }
    return cloud;
  }

  visualization_msgs::msg::Marker velocity_marker(
    const int id, const std::string & label, const double x, const double y, const float red, const float green, const float blue) const {
    visualization_msgs::msg::Marker marker;
    marker.header.stamp = now(); marker.header.frame_id = "base_link"; marker.ns = "cbf_velocity"; marker.id = id;
    marker.type = visualization_msgs::msg::Marker::ARROW; marker.action = visualization_msgs::msg::Marker::ADD;
    marker.scale.x = 0.04; marker.scale.y = 0.09; marker.scale.z = 0.12; marker.color.a = 1.0F;
    marker.color.r = red; marker.color.g = green; marker.color.b = blue;
    geometry_msgs::msg::Point start, end; end.x = x; end.y = y; marker.points = {start, end}; marker.text = label;
    marker.lifetime = rclcpp::Duration::from_seconds(0.25); return marker;
  }

  void publish_debug() {
    DebugState copy;
    { std::lock_guard<std::mutex> lock(debug_mutex_); copy = debug_; }
    candidate_pub_->publish(make_cloud(copy.candidates.data(), copy.candidate_count, 0xFF33CCFF));
    selected_pub_->publish(make_cloud(copy.selected.data(), copy.selected_count,
      copy.result.max_slack > 0.0 ? 0xFF2222FF : 0xFFFFA500));
    visualization_msgs::msg::MarkerArray markers;
    markers.markers.push_back(velocity_marker(0, "measured", copy.measured_x, copy.measured_y, 1.0F, 1.0F, 1.0F));
    markers.markers.push_back(velocity_marker(1, "policy", copy.policy_x, copy.policy_y, 0.2F, 0.5F, 1.0F));
    markers.markers.push_back(velocity_marker(2, "nominal",
      copy.measured_x + copy.nominal_x / (copy.in_goal_region ? config_.goal_region_kp : config_.kp_x),
      copy.measured_y + copy.nominal_y / (copy.in_goal_region ? config_.goal_region_kp : config_.kp_y),
      1.0F, 1.0F, 0.0F));
    markers.markers.push_back(velocity_marker(3, copy.fallback ? "fallback" : "safe", copy.command_x, copy.command_y,
      copy.fallback ? 1.0F : 0.0F, copy.fallback ? 0.0F : 1.0F, 0.0F));
    marker_pub_->publish(markers);
    diagnostic_msgs::msg::DiagnosticStatus status;
    status.name = "go2_cbf_control";
    status.hardware_id = "go2";
    status.level = copy.fallback ? diagnostic_msgs::msg::DiagnosticStatus::WARN : diagnostic_msgs::msg::DiagnosticStatus::OK;
    status.message = copy.fallback ? std::string("controlled_stop: ") + copy.fallback_reason : copy.result.status_text;
    auto add = [&status](const std::string & key, const double value) {
      diagnostic_msgs::msg::KeyValue item; item.key = key; item.value = std::to_string(value); status.values.push_back(item);
    };
    auto add_text = [&status](const std::string & key, const char * value) {
      diagnostic_msgs::msg::KeyValue item; item.key = key; item.value = value; status.values.push_back(item);
    };
    add_text("solver_status", copy.result.status_text); add_text("fallback_reason", copy.fallback_reason);
    add("policy_age_s", copy.policy_age); add("velocity_age_s", copy.velocity_age); add("scan_age_s", copy.scan_age);
    add("cbf_margin_m", copy.margin);
    add("in_goal_region", copy.in_goal_region ? 1.0 : 0.0);
    add("goal_region_kp", config_.goal_region_kp);
    add("effective_kp_x", copy.in_goal_region ? config_.goal_region_kp : config_.kp_x);
    add("effective_kp_y", copy.in_goal_region ? config_.goal_region_kp : config_.kp_y);
    add("solve_time_s", copy.result.solve_time_s); add("control_elapsed_s", copy.elapsed_s);
    add("qp_update_time_s", copy.result.update_time_s); add("timer_lateness_s", copy.timer_lateness_s);
    add("release_to_publish_s", copy.release_to_publish_s); add("iterations", copy.result.iterations);
    add("primal_residual", copy.result.primal_residual); add("dual_residual", copy.result.dual_residual);
    add("max_slack", copy.result.max_slack); add("candidate_count", static_cast<double>(copy.candidate_count));
    add("selected_count", static_cast<double>(copy.selected_count)); add("timeout_count", copy.timeout_count);
    add("fallback_transitions", copy.fallback_transitions);
    diagnostic_msgs::msg::DiagnosticArray array; array.header.stamp = now(); array.status.push_back(std::move(status)); status_pub_->publish(array);
  }

  struct DebugState {
    double policy_age{std::numeric_limits<double>::infinity()}, velocity_age{std::numeric_limits<double>::infinity()};
    double scan_age{std::numeric_limits<double>::infinity()}, margin{}, policy_x{}, policy_y{}, nominal_x{}, nominal_y{}, measured_x{}, measured_y{};
    bool in_goal_region{};
    double command_x{}, command_y{}, command_wz{}, elapsed_s{}, timer_lateness_s{}, release_to_publish_s{};
    uint64_t timeout_count{}, fallback_transitions{};
    SolverResult result{}; bool fallback{true}; const char * fallback_reason{"not_run"};
    std::array<Point, kScanBins> candidates{}; int candidate_count{};
    std::array<Point, kMaxPoints> selected{}; int selected_count{};
  };

  Config config_;
  StaticCbfQp qp_;
  rclcpp::CallbackGroup::SharedPtr io_group_, control_group_, debug_group_;
  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr policy_sub_, velocity_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr goal_region_sub_;
  rclcpp::Subscription<go2_dds_ros2_bridge_msgs::msg::CbfScan>::SharedPtr scan_sub_;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr command_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr status_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr candidate_pub_, selected_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::TimerBase::SharedPtr control_timer_, debug_timer_;
  mutable std::mutex policy_mutex_, velocity_mutex_, scan_mutex_, goal_region_mutex_, debug_mutex_;
  std::shared_ptr<geometry_msgs::msg::TwistStamped> policy_, velocity_;
  std::shared_ptr<go2_dds_ros2_bridge_msgs::msg::CbfScan> scan_;
  std::optional<bool> in_goal_region_;
  Clock::time_point last_control_{};
  double last_command_x_{}, last_command_y_{}, last_command_wz_{};
  bool fallback_active_{true}; int healthy_cycles_{}; uint64_t timeout_count_{}, fallback_transitions_{}; DebugState debug_{};
};

}  // namespace

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<CbfControlNode>();
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 3);
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
