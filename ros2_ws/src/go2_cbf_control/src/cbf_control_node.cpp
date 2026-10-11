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

#include <go2_dds_ros2_bridge_msgs/msg/cbf_control_snapshot.hpp>
#include <go2_dds_ros2_bridge_msgs/msg/cbf_world_scan.hpp>
#include <osqp.h>

#include "go2_cbf_control/cbf_math.hpp"

#include "go2_cbf_control/cbf_qp.hpp"

namespace {
using namespace go2_cbf_control;

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
    scan_sub_ = create_subscription<go2_dds_ros2_bridge_msgs::msg::CbfControlSnapshot>(
      config_.scan_topic, qos, [this](go2_dds_ros2_bridge_msgs::msg::CbfControlSnapshot::SharedPtr message) {
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
    config.scan_topic = declare_parameter<std::string>("scan_topic", "/cbf/snapshot");
    config.cmd_topic = declare_parameter<std::string>("cmd_topic", "/cmd_vel");
    config.control_hz = declare_parameter<double>("control_hz", 50.0);
    config.solver_time_limit_s = declare_parameter<double>("solver_time_limit_s", 0.005);
    config.publish_deadline_s = declare_parameter<double>("publish_deadline_s", 0.010);
    config.policy_timeout_s = declare_parameter<double>("policy_timeout_s", 0.25);
    config.velocity_timeout_s = declare_parameter<double>("state_timeout_s", 0.10);
    config.max_interpolation_gap_s = declare_parameter<double>("max_interpolation_gap_s", 0.10);
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
    config.max_yaw_rate_radps = declare_parameter<double>("max_yaw_rate_radps", 0.6);
    config.max_yaw_accel_radps2 = declare_parameter<double>("max_yaw_accel_radps2", 1.0);
    config.enable_navigation_slew = declare_parameter<bool>("enable_navigation_slew", true);
    config.a_nav_mps2 = declare_parameter<double>("a_nav", 2.0);
    config.planar_deadzone_mps = declare_parameter<double>("planar_deadzone_mps", 0.1);
    config.yaw_deadzone_radps = declare_parameter<double>("yaw_deadzone_radps", 0.1);
    config.tracking_tau_s = declare_parameter<double>("tracking_tau_s", 0.30);
    config.max_lidar_points = declare_parameter<int>("max_lidar_points", 64);
    config.slack_penalty = declare_parameter<double>("slack_penalty", 1000.0);
    config.max_cbf_slack = declare_parameter<double>("max_cbf_slack", 0.05);
    config.bad_solve_grace_s = declare_parameter<double>("bad_solve_grace_s", 0.25);
    config.fallback_linear_decel_mps2 = declare_parameter<double>("fallback_linear_decel_mps2", 0.5);
    config.fallback_yaw_decel_radps2 = declare_parameter<double>("fallback_yaw_decel_radps2", 1.0);
    if (config.velocity_timeout_s <= 0.0 || config.max_interpolation_gap_s <= 0.0 || config.scan_timeout_s <= 0.0 || config.control_hz <= 0.0 || config.solver_time_limit_s <= 0.0 || config.publish_deadline_s <= config.solver_time_limit_s ||
        config.tracking_tau_s <= 0.0 || config.d_margin_m <= 0.0 || config.d_cbf_active_m < config.d_margin_m ||
        config.gamma1 <= 0.0 || config.gamma2 <= 0.0 || config.kp_x <= 0.0 || config.kp_y <= 0.0 ||
        config.goal_region_kp <= 0.0 ||
        config.accel_limit_x <= 0.0 || config.accel_limit_y <= 0.0 || config.velocity_limit_x <= 0.0 ||
        config.velocity_limit_y <= 0.0 || config.max_yaw_rate_radps <= 0.0 ||
        config.max_yaw_accel_radps2 <= 0.0 || config.slack_penalty <= 0.0 ||
        config.a_nav_mps2 <= 0.0 ||
        config.planar_deadzone_mps < 0.0 || config.yaw_deadzone_radps < 0.0 ||
        config.max_cbf_slack < 0.0 || config.bad_solve_grace_s <= 0.0 ||
        config.goal_region_topic.empty()) {
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

  bool try_copy_scan(std::shared_ptr<go2_dds_ros2_bridge_msgs::msg::CbfControlSnapshot> & destination) {
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
    const go2_dds_ros2_bridge_msgs::msg::CbfWorldScan & scan,
    const double robot_x, const double robot_y,
    std::array<Point, kMaxPoints> & selected, int & selected_count,
    std::array<Point, kCandidateBins> & candidates, int & candidate_count,
    int & front_candidate_count, int & static_candidate_count) const
  {
    candidate_count = front_candidate_count = static_candidate_count = 0;
    const auto append_hit = [&](const double x_value, const double y_value, const uint8_t hit, const bool is_static) {
      if (hit > 1) return false;
      if (hit == 0) return true;
      const double x = x_value - robot_x;
      const double y = y_value - robot_y;
      const double range = std::hypot(x, y);
      if (!std::isfinite(x) || !std::isfinite(y) || range <= 1.0e-4 || range > config_.d_cbf_active_m) return true;
      candidates[candidate_count++] = {x, y, range, is_static};
      if (is_static) ++static_candidate_count;
      else ++front_candidate_count;
      return true;
    };
    for (int index = 0; index < kFrontScanBins; ++index) {
      if (!append_hit(scan.points_xy_m[2 * index], scan.points_xy_m[2 * index + 1], scan.hits[index], false)) return false;
    }
    for (int index = 0; index < kStaticScanBins; ++index) {
      if (!append_hit(
          scan.static_points_xy_m[2 * index], scan.static_points_xy_m[2 * index + 1], scan.static_hits[index], true)) return false;
    }
    std::sort(candidates.begin(), candidates.begin() + candidate_count, [](const Point & left, const Point & right) {
      return left.range < right.range;
    });
    selected_count = std::min(candidate_count, kMaxPoints);
    for (int index = 0; index < selected_count; ++index) selected[index] = candidates[index];
    return true;
  }

  // Solved and time-limited iterates must satisfy the actual unscaled QP rows.
  // Output deadzones/fallback happen later and are not certified by this check.
  const char * normal_solution_validation_failure(const SolverResult & result, const int count) const
  {
    return validate_candidate(result, count, config_.max_cbf_slack);
  }

  void control_step() {
    const auto start = Clock::now();
    const double expected_step_s = 1.0 / config_.control_hz;
    const double step_s = std::max(0.0, std::chrono::duration<double>(start - last_control_).count());
    const double timer_lateness_s = std::max(0.0, step_s - expected_step_s);
    last_control_ = start;
    std::shared_ptr<geometry_msgs::msg::TwistStamped> policy;
    std::shared_ptr<go2_dds_ros2_bridge_msgs::msg::CbfControlSnapshot> scan;
    const bool in_goal_region = try_copy_goal_region().value_or(false);
    const bool inputs_available = try_copy_policy(policy) && try_copy_scan(scan);
    const double policy_age = policy ? ros_age_s(policy->header.stamp) : std::numeric_limits<double>::infinity();
    const double velocity_age = scan ? ros_age_s(scan->header.stamp) : std::numeric_limits<double>::infinity();
    const double scan_age = scan ? ros_age_s(scan->scan.scan_start) : std::numeric_limits<double>::infinity();
    const bool valid_headers = policy && scan && scan->valid && policy->header.frame_id == "base_link" &&
      scan->header.frame_id == "cbf_world" && scan->scan.header.frame_id == "cbf_world" &&
      !scan->session_id.empty() && scan->session_id == scan->scan.session_id &&
      rclcpp::Time(scan->header.stamp).nanoseconds() <= now().nanoseconds() &&
      rclcpp::Time(scan->header.stamp).nanoseconds() >= rclcpp::Time(scan->scan.header.stamp).nanoseconds() &&
      rclcpp::Time(scan->scan.header.stamp).nanoseconds() >= rclcpp::Time(scan->scan.scan_start).nanoseconds() &&
      std::isfinite(scan->robot_yaw) && std::isfinite(scan->robot_xy_m[0]) && std::isfinite(scan->robot_xy_m[1]) &&
      std::isfinite(scan->interpolation_gap_s) && scan->interpolation_gap_s >= 0.0 && scan->interpolation_gap_s <= config_.max_interpolation_gap_s;
    if (scan && scan->session_id != active_session_) {
      active_session_ = scan->session_id;
      last_command_x_ = last_command_y_ = last_command_wz_ = 0.0;
      filtered_policy_x_ = filtered_policy_y_ = yaw_governor_state_wz_ = 0.0;
    }
    const double yaw = scan ? scan->robot_yaw : 0.0;
    const double cy = std::cos(yaw), sy = std::sin(yaw);
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
    std::array<Point, kCandidateBins> candidates{};
    int candidate_count = 0;
    int front_candidate_count = 0;
    int static_candidate_count = 0;
    SolverResult result;
    double nominal_x = 0.0, nominal_y = 0.0, margin = config_.d_margin_m;
    double policy_target_x = filtered_policy_x_, policy_target_y = filtered_policy_y_;
    double measured_x = 0.0, measured_y = 0.0, command_x = 0.0, command_y = 0.0, command_wz = 0.0;
    if (healthy) {
      measured_x = cy * scan->velocity_xy_mps[0] + sy * scan->velocity_xy_mps[1];
      measured_y = -sy * scan->velocity_xy_mps[0] + cy * scan->velocity_xy_mps[1];
      if (!std::isfinite(measured_x) || !std::isfinite(measured_y) ||
          !std::isfinite(policy->twist.linear.x) || !std::isfinite(policy->twist.linear.y) ||
          !std::isfinite(policy->twist.angular.z)) {
        healthy = false;
        fallback_reason = "nonfinite_input";
      }
    }
    if (healthy) {
      healthy = select_points(
        scan->scan, scan->robot_xy_m[0], scan->robot_xy_m[1], selected, selected_count, candidates, candidate_count, front_candidate_count, static_candidate_count);
      if (!healthy) fallback_reason = "malformed_scan";
    }
    if (healthy) {
      const double zoh = go2_cbf_control::zoh_gain(step_s, config_.tracking_tau_s);
      const double kp_x = in_goal_region ? config_.goal_region_kp : config_.kp_x;
      const double kp_y = in_goal_region ? config_.goal_region_kp : config_.kp_y;
      // Non-standard deployment heuristic: slew only the planar navigation
      // reference, before the CBF QP. This does not change CBF constraints or
      // guarantee a slew bound on /cmd_vel. The filter advances only for an
      // accepted control tick, and does not alter yaw or raw policy diagnostics.
      const auto target = go2_cbf_control::planar_policy_reference(
        config_.enable_navigation_slew, policy->twist.linear.x, policy->twist.linear.y,
        filtered_policy_x_, filtered_policy_y_, config_.a_nav_mps2, step_s);
      policy_target_x = target[0];
      policy_target_y = target[1];
      nominal_x = std::clamp(kp_x * (policy_target_x - measured_x), -config_.accel_limit_x, config_.accel_limit_x);
      nominal_y = std::clamp(kp_y * (policy_target_y - measured_y), -config_.accel_limit_y, config_.accel_limit_y);
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
        result = qp_.solve(cy * nominal_x - sy * nominal_y, sy * nominal_x + cy * nominal_y,
          selected, selected_count, scan->velocity_xy_mps[0], scan->velocity_xy_mps[1], margin,
          lower_x, lower_y, upper_x, upper_y, yaw);
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
          command_x = std::clamp(measured_x + zoh * (cy * result.u_x + sy * result.u_y), -config_.velocity_limit_x, config_.velocity_limit_x);
          command_y = std::clamp(measured_y + zoh * (-sy * result.u_x + cy * result.u_y), -config_.velocity_limit_y, config_.velocity_limit_y);
        }
      }
      // The planar CBF-QP does not constrain yaw.  Shape yaw here at the
      // control rate so a new policy target cannot cause either an excessive
      // yaw rate or an abrupt angular-command step at /cmd_vel.
      command_wz = go2_cbf_control::limited_yaw_command(
        policy->twist.angular.z, yaw_governor_state_wz_, config_.max_yaw_rate_radps,
        config_.max_yaw_accel_radps2, step_s);
    }
    const double elapsed_s = std::chrono::duration<double>(Clock::now() - start).count();
    if (elapsed_s > config_.publish_deadline_s) {
      healthy = false;
      fallback_reason = "control_deadline_missed";
    }
    if (result.timed_out_or_iter_limit) ++timeout_count_;
    double bad_solve_duration_s = 0.0;
    bool holding_last_valid_command = false;
    if (healthy) {
      // An accepted result immediately replaces a held or braking command.
      // Do not latch controlled stop: a later, fresh CBF solution is more
      // informed than the command from the failed tick.
      bad_solve_started_at_.reset();
      if (fallback_active_) ++fallback_transitions_;
      fallback_active_ = false;
      filtered_policy_x_ = policy_target_x;
      filtered_policy_y_ = policy_target_y;
    } else {
      if (!bad_solve_started_at_) bad_solve_started_at_ = start;
      bad_solve_duration_s = std::max(
        0.0, std::chrono::duration<double>(start - *bad_solve_started_at_).count());
      const auto response = go2_cbf_control::fault_response(
        false, bad_solve_duration_s, config_.bad_solve_grace_s);
      if (response == go2_cbf_control::FaultResponse::kHoldLastValidCommand) {
        // A transient rejected solve must not invent a braking command. Keep
        // republishing the last accepted command until the grace period ends.
        holding_last_valid_command = true;
        command_x = last_command_x_;
        command_y = last_command_y_;
        command_wz = last_command_wz_;
      } else {
        if (!fallback_active_) ++fallback_transitions_;
        fallback_active_ = true;
        // Restart the reference from rest after a sustained input/solve fault.
        filtered_policy_x_ = 0.0;
        filtered_policy_y_ = 0.0;
        command_x = go2_cbf_control::approach_zero(last_command_x_, config_.fallback_linear_decel_mps2 * step_s);
        command_y = go2_cbf_control::approach_zero(last_command_y_, config_.fallback_linear_decel_mps2 * step_s);
        command_wz = go2_cbf_control::approach_zero(last_command_wz_, config_.fallback_yaw_decel_radps2 * step_s);
      }
    }
    // Keep the pre-deadzone yaw state so sub-threshold ramp steps accumulate.
    // A short fault hold retains that state; a controlled stop resets it to
    // the fallback command, avoiding a stale yaw jump when solves resume.
    if (!holding_last_valid_command) yaw_governor_state_wz_ = command_wz;
    // Apply once to the completed CBF/fallback command. The published command,
    // remembered command, and diagnostics must all agree.
    go2_cbf_control::apply_velocity_deadzone(
      command_x, command_y, command_wz, config_.planar_deadzone_mps, config_.yaw_deadzone_radps);
    const auto publish_start = Clock::now();
    publish_command(command_x, command_y, command_wz);
    const double release_to_publish_s = std::chrono::duration<double>(Clock::now() - publish_start).count();
    last_command_x_ = command_x;
    last_command_y_ = command_y;
    last_command_wz_ = command_wz;
    if (scan) {
      for (int i = 0; i < candidate_count; ++i) { candidates[i].x += scan->robot_xy_m[0]; candidates[i].y += scan->robot_xy_m[1]; }
      for (int i = 0; i < selected_count; ++i) { selected[i].x += scan->robot_xy_m[0]; selected[i].y += scan->robot_xy_m[1]; }

    }
    update_debug(policy_age, velocity_age, scan_age, margin, in_goal_region, policy ? policy->twist.linear.x : 0.0,
      policy ? policy->twist.linear.y : 0.0, policy ? policy->twist.angular.z : 0.0,
      filtered_policy_x_, filtered_policy_y_,
      nominal_x, nominal_y, measured_x, measured_y,
      command_x, command_y, command_wz, elapsed_s, timer_lateness_s, release_to_publish_s, result,
      candidates, candidate_count, front_candidate_count, static_candidate_count, selected, selected_count,
      timeout_count_, fallback_transitions_,
      bad_solve_duration_s, holding_last_valid_command, fallback_reason, scan.get());
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
    double policy_x, double policy_y, double policy_wz, double filtered_policy_x, double filtered_policy_y,
    double nominal_x, double nominal_y,
    double measured_x, double measured_y, double command_x, double command_y, double command_wz, double elapsed_s,
    double timer_lateness_s, double release_to_publish_s, const SolverResult & result,
    const std::array<Point, kCandidateBins> & candidates, int candidate_count,
    int front_candidate_count, int static_candidate_count,
    const std::array<Point, kMaxPoints> & selected, int selected_count,
    uint64_t timeout_count, uint64_t fallback_transitions, double bad_solve_duration_s,
    bool holding_last_valid_command, const char * fallback_reason,
    const go2_dds_ros2_bridge_msgs::msg::CbfControlSnapshot * snapshot)
  {
    std::unique_lock<std::mutex> lock(debug_mutex_, std::try_to_lock);
    if (!lock.owns_lock()) return;
    if (snapshot) {
      debug_.evaluation_stamp = snapshot->header.stamp;
      debug_.session = snapshot->session_id;
      debug_.alignment_reason = snapshot->reason;
      debug_.interpolation_gap = snapshot->interpolation_gap_s;
      debug_.robot_x = snapshot->robot_xy_m[0]; debug_.robot_y = snapshot->robot_xy_m[1]; debug_.yaw = snapshot->robot_yaw;
    }
    debug_.policy_age = policy_age; debug_.velocity_age = velocity_age; debug_.scan_age = scan_age; debug_.margin = margin;
    debug_.in_goal_region = in_goal_region;
    debug_.policy_x = policy_x; debug_.policy_y = policy_y; debug_.policy_wz = policy_wz;
    debug_.filtered_policy_x = filtered_policy_x; debug_.filtered_policy_y = filtered_policy_y;
    debug_.nominal_x = nominal_x; debug_.nominal_y = nominal_y; debug_.measured_x = measured_x; debug_.measured_y = measured_y;
    debug_.command_x = command_x; debug_.command_y = command_y; debug_.command_wz = command_wz; debug_.elapsed_s = elapsed_s;
    debug_.timer_lateness_s = timer_lateness_s; debug_.release_to_publish_s = release_to_publish_s;
    debug_.timeout_count = timeout_count; debug_.fallback_transitions = fallback_transitions;
    debug_.bad_solve_duration_s = bad_solve_duration_s;
    debug_.result = result; debug_.fallback = fallback_active_;
    debug_.holding_last_valid_command = holding_last_valid_command;
    debug_.fallback_reason = fallback_reason; debug_.candidates = candidates;
    debug_.candidate_count = candidate_count;
    debug_.front_candidate_count = front_candidate_count;
    debug_.static_candidate_count = static_candidate_count;
    debug_.selected = selected; debug_.selected_count = selected_count;
  }

  sensor_msgs::msg::PointCloud2 make_cloud(const Point * points, const std::size_t point_count, const uint32_t rgba, const builtin_interfaces::msg::Time & stamp) const {
    sensor_msgs::msg::PointCloud2 cloud;
    cloud.header.stamp = stamp; cloud.header.frame_id = "cbf_world";
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
    candidate_pub_->publish(make_cloud(copy.candidates.data(), copy.candidate_count, 0xFF33CCFF, copy.evaluation_stamp));
    selected_pub_->publish(make_cloud(copy.selected.data(), copy.selected_count,
      copy.result.max_slack > 0.0 ? 0xFF2222FF : 0xFFFFA500, copy.evaluation_stamp));
    visualization_msgs::msg::MarkerArray markers;
    markers.markers.push_back(velocity_marker(0, "measured", copy.measured_x, copy.measured_y, 1.0F, 1.0F, 1.0F));
    markers.markers.push_back(velocity_marker(1, "policy", copy.policy_x, copy.policy_y, 0.2F, 0.5F, 1.0F));
    markers.markers.push_back(velocity_marker(4, "slewed_policy", copy.filtered_policy_x, copy.filtered_policy_y,
      0.7F, 0.3F, 1.0F));
    markers.markers.push_back(velocity_marker(2, "nominal",
      copy.measured_x + copy.nominal_x / (copy.in_goal_region ? config_.goal_region_kp : config_.kp_x),
      copy.measured_y + copy.nominal_y / (copy.in_goal_region ? config_.goal_region_kp : config_.kp_y),
      1.0F, 1.0F, 0.0F));
    const bool degraded = copy.fallback || copy.holding_last_valid_command;
    const char * command_label = copy.fallback ? "controlled_stop" :
      (copy.holding_last_valid_command ? "holding_last_valid" : "safe");
    markers.markers.push_back(velocity_marker(3, command_label, copy.command_x, copy.command_y,
      degraded ? 1.0F : 0.0F, degraded ? 0.0F : 1.0F, 0.0F));
    // Express command-axis arrows at the same evaluated pose/time as the points.
    for (auto & marker : markers.markers) {
      const double vx = marker.points[1].x, vy = marker.points[1].y;
      marker.header.frame_id = "cbf_world";
      marker.header.stamp = copy.evaluation_stamp;
      marker.points[0].x = copy.robot_x; marker.points[0].y = copy.robot_y;
      marker.points[1].x = copy.robot_x + std::cos(copy.yaw)*vx - std::sin(copy.yaw)*vy;
      marker.points[1].y = copy.robot_y + std::sin(copy.yaw)*vx + std::cos(copy.yaw)*vy;
    }
    marker_pub_->publish(markers);
    diagnostic_msgs::msg::DiagnosticStatus status;
    status.name = "go2_cbf_control";
    status.hardware_id = "go2";
    status.level = degraded ? diagnostic_msgs::msg::DiagnosticStatus::WARN : diagnostic_msgs::msg::DiagnosticStatus::OK;
    status.message = copy.fallback ? std::string("controlled_stop: ") + copy.fallback_reason :
      (copy.holding_last_valid_command ? std::string("holding_last_valid_command: ") + copy.fallback_reason :
      copy.result.status_text);
    auto add = [&status](const std::string & key, const double value) {
      diagnostic_msgs::msg::KeyValue item; item.key = key; item.value = std::to_string(value); status.values.push_back(item);
    };
    auto add_text = [&status](const std::string & key, const char * value) {
      diagnostic_msgs::msg::KeyValue item; item.key = key; item.value = value; status.values.push_back(item);
    };
    add_text("solver_status", copy.result.status_text.c_str()); add_text("fallback_reason", copy.fallback_reason);
    add("policy_age_s", copy.policy_age); add("velocity_age_s", copy.velocity_age); add("scan_age_s", copy.scan_age);
    add("cbf_margin_m", copy.margin);
    add("in_goal_region", copy.in_goal_region ? 1.0 : 0.0);
    add("goal_region_kp", config_.goal_region_kp);
    add("effective_kp_x", copy.in_goal_region ? config_.goal_region_kp : config_.kp_x);
    add("effective_kp_y", copy.in_goal_region ? config_.goal_region_kp : config_.kp_y);
    add("policy_wz_radps", copy.policy_wz); add("command_wz_radps", copy.command_wz);
    add("navigation_slew_enabled", config_.enable_navigation_slew ? 1.0 : 0.0);
    add("a_nav_mps2", config_.a_nav_mps2);
    add("filtered_policy_x_mps", copy.filtered_policy_x);
    add("filtered_policy_y_mps", copy.filtered_policy_y);
    add("max_yaw_rate_radps", config_.max_yaw_rate_radps);
    add("max_yaw_accel_radps2", config_.max_yaw_accel_radps2);
    add("solve_time_s", copy.result.solve_time_s); add("control_elapsed_s", copy.elapsed_s);
    add("qp_update_time_s", copy.result.update_time_s); add("timer_lateness_s", copy.timer_lateness_s);
    add("release_to_publish_s", copy.release_to_publish_s); add("iterations", copy.result.iterations);
    add("primal_residual", copy.result.primal_residual); add("dual_residual", copy.result.dual_residual);
    add("max_constraint_violation", copy.result.max_constraint_violation);
    add("state_age_s", copy.velocity_age);
    add("interpolation_gap_s", copy.interpolation_gap);
    add("evaluation_time_s", rclcpp::Time(copy.evaluation_stamp).seconds());
    add_text("world_session", copy.session.c_str());
    add_text("alignment_reason", copy.alignment_reason.c_str());
    add_text("constraint_check_scope", "QP_candidate_before_deadzone_and_fallback");
    add("max_slack", copy.result.max_slack); add("candidate_count", static_cast<double>(copy.candidate_count));
    add("front_candidate_count", static_cast<double>(copy.front_candidate_count));
    add("static_candidate_count", static_cast<double>(copy.static_candidate_count));
    const auto selected_static_count = static_cast<double>(std::count_if(
      copy.selected.begin(), copy.selected.begin() + copy.selected_count,
      [](const Point & point) { return point.is_static; }));
    add("selected_static_count", selected_static_count);
    add("selected_front_count", static_cast<double>(copy.selected_count) - selected_static_count);
    add("selected_count", static_cast<double>(copy.selected_count)); add("timeout_count", copy.timeout_count);
    add("fallback_transitions", copy.fallback_transitions);
    add("bad_solve_duration_s", copy.bad_solve_duration_s);
    diagnostic_msgs::msg::DiagnosticArray array; array.header.stamp = now(); array.status.push_back(std::move(status)); status_pub_->publish(array);
  }

  struct DebugState {
    builtin_interfaces::msg::Time evaluation_stamp;
    std::string session, alignment_reason;
    double interpolation_gap{}, robot_x{}, robot_y{}, yaw{};
    double policy_age{std::numeric_limits<double>::infinity()}, velocity_age{std::numeric_limits<double>::infinity()};
    double scan_age{std::numeric_limits<double>::infinity()}, margin{}, policy_x{}, policy_y{}, policy_wz{}, filtered_policy_x{}, filtered_policy_y{}, nominal_x{}, nominal_y{}, measured_x{}, measured_y{};
    bool in_goal_region{};
    double command_x{}, command_y{}, command_wz{}, elapsed_s{}, timer_lateness_s{}, release_to_publish_s{};
    uint64_t timeout_count{}, fallback_transitions{};
    double bad_solve_duration_s{};
    SolverResult result{}; bool fallback{}; bool holding_last_valid_command{};
    const char * fallback_reason{"not_run"};
    std::array<Point, kCandidateBins> candidates{}; int candidate_count{}, front_candidate_count{}, static_candidate_count{};
    std::array<Point, kMaxPoints> selected{}; int selected_count{};
  };

  std::string active_session_;
  Config config_;
  StaticCbfQp qp_;
  rclcpp::CallbackGroup::SharedPtr io_group_, control_group_, debug_group_;
  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr policy_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr goal_region_sub_;
  rclcpp::Subscription<go2_dds_ros2_bridge_msgs::msg::CbfControlSnapshot>::SharedPtr scan_sub_;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr command_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr status_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr candidate_pub_, selected_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::TimerBase::SharedPtr control_timer_, debug_timer_;
  mutable std::mutex policy_mutex_, scan_mutex_, goal_region_mutex_, debug_mutex_;
  std::shared_ptr<geometry_msgs::msg::TwistStamped> policy_;
  std::shared_ptr<go2_dds_ros2_bridge_msgs::msg::CbfControlSnapshot> scan_;
  std::optional<bool> in_goal_region_;
  Clock::time_point last_control_{};
  double last_command_x_{}, last_command_y_{}, last_command_wz_{};
  double filtered_policy_x_{}, filtered_policy_y_{};
  double yaw_governor_state_wz_{};
  bool fallback_active_{};
  std::optional<Clock::time_point> bad_solve_started_at_;
  uint64_t timeout_count_{}, fallback_transitions_{};
  DebugState debug_;
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
