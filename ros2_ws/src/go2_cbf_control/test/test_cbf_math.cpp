#include <gtest/gtest.h>

#include "go2_cbf_control/cbf_math.hpp"

TEST(CbfMath, ZohGainIsPositiveAndExceedsStep) {
  EXPECT_GT(go2_cbf_control::zoh_gain(0.02, 0.30), 0.02);
}

TEST(CbfMath, StaticBarrierMatchesReferenceExpression) {
  EXPECT_DOUBLE_EQ(
    go2_cbf_control::static_barrier_offset(-1.0, 0.0, 0.5, 0.0, 2.0, 2.0, 0.7),
    2.0 * 0.25 + 2.0 * 4.0 * -0.5 + 4.0 * (1.0 - 0.49));
}

TEST(CbfMath, ControlledStopNeverChangesSign) {
  EXPECT_DOUBLE_EQ(go2_cbf_control::approach_zero(0.01, 0.02), 0.0);
  EXPECT_DOUBLE_EQ(go2_cbf_control::approach_zero(-0.4, 0.1), -0.3);
}

TEST(CbfMath, PlanarTargetSlewLimitsVectorChangeWithoutDistortingDirection) {
  const auto first = go2_cbf_control::slew_planar_target(0.6, 0.8, 0.0, 0.0, 2.0, 0.02);
  EXPECT_NEAR(first[0], 0.024, 1e-12);
  EXPECT_NEAR(first[1], 0.032, 1e-12);
  EXPECT_NEAR(std::hypot(first[0], first[1]), 0.04, 1e-12);

  const auto small = go2_cbf_control::slew_planar_target(0.03, 0.02, 0.0, 0.0, 2.0, 0.02);
  EXPECT_DOUBLE_EQ(small[0], 0.03);
  EXPECT_DOUBLE_EQ(small[1], 0.02);
  const auto no_time = go2_cbf_control::slew_planar_target(1.0, 0.0, 0.0, 0.0, 2.0, 0.0);
  EXPECT_DOUBLE_EQ(no_time[0], 0.0);
}

TEST(CbfMath, PlanarTargetSlewReversesAndConvergesWithoutOvershoot) {
  double x = 0.4, y = 0.0;
  for (int tick = 0; tick < 20; ++tick) {
    const auto next = go2_cbf_control::slew_planar_target(-0.2, 0.0, x, y, 2.0, 0.02);
    EXPECT_LE(std::hypot(next[0] - x, next[1] - y), 0.04 + 1e-12);
    x = next[0]; y = next[1];
  }
  EXPECT_NEAR(x, -0.2, 1e-12);
  EXPECT_DOUBLE_EQ(y, 0.0);
}

TEST(CbfMath, DisabledNavigationSlewPassesPolicyReferenceUnchanged) {
  const auto target = go2_cbf_control::planar_policy_reference(
    false, 1.0, -0.5, 0.0, 0.0, 2.0, 0.02);
  EXPECT_DOUBLE_EQ(target[0], 1.0);
  EXPECT_DOUBLE_EQ(target[1], -0.5);
  const auto enabled = go2_cbf_control::planar_policy_reference(
    true, 1.0, -0.5, 0.0, 0.0, 2.0, 0.02);
  EXPECT_NEAR(std::hypot(enabled[0], enabled[1]), 0.04, 1e-12);
}

TEST(CbfMath, VelocityDeadzoneGatesPlanarNormAndYawIndependently) {
  double x = 0.06, y = 0.07, wz = 0.2;
  go2_cbf_control::apply_velocity_deadzone(x, y, wz, 0.1, 0.1);
  EXPECT_DOUBLE_EQ(x, 0.0);
  EXPECT_DOUBLE_EQ(y, 0.0);
  EXPECT_DOUBLE_EQ(wz, 0.2);

  x = 0.08; y = 0.08; wz = -0.09;
  go2_cbf_control::apply_velocity_deadzone(x, y, wz, 0.1, 0.1);
  EXPECT_DOUBLE_EQ(x, 0.08);
  EXPECT_DOUBLE_EQ(y, 0.08);
  EXPECT_DOUBLE_EQ(wz, 0.0);

  x = 0.1; y = 0.0; wz = -0.1;
  go2_cbf_control::apply_velocity_deadzone(x, y, wz, 0.1, 0.1);
  EXPECT_DOUBLE_EQ(x, 0.1);
  EXPECT_DOUBLE_EQ(wz, -0.1);
}

TEST(CbfMath, YawGovernorLimitsRateAndAcceleration) {
  // At 50 Hz and 1 rad/s^2, a target may alter the command by 0.02 rad/s.
  EXPECT_DOUBLE_EQ(go2_cbf_control::limited_yaw_command(2.0, 0.0, 0.6, 1.0, 0.02), 0.02);
  EXPECT_DOUBLE_EQ(go2_cbf_control::limited_yaw_command(-2.0, 0.6, 0.6, 1.0, 0.02), 0.58);
  EXPECT_DOUBLE_EQ(go2_cbf_control::limited_yaw_command(2.0, 2.0, 0.6, 1.0, 0.02), 0.6);
}

TEST(CbfMath, YawGovernorAccumulatesThroughOutputDeadzone) {
  // At 50 Hz, each 3 rad/s^2 step is only 0.06 rad/s. Applying the
  // 0.10-rad/s deadzone to the governor state would prevent any turn.
  double governor_state = 0.0;
  double published_yaw = 0.0;
  for (int tick = 0; tick < 2; ++tick) {
    governor_state = go2_cbf_control::limited_yaw_command(1.0, governor_state, 2.0, 3.0, 0.02);
    published_yaw = governor_state;
    double x = 0.0, y = 0.0;
    go2_cbf_control::apply_velocity_deadzone(x, y, published_yaw, 0.1, 0.1);
    if (tick == 0) EXPECT_DOUBLE_EQ(published_yaw, 0.0);
  }
  EXPECT_NEAR(governor_state, 0.12, 1e-12);
  EXPECT_NEAR(published_yaw, 0.12, 1e-12);
}

TEST(CbfMath, TransientBadTicksHoldTheLastValidCommand) {
  constexpr double grace_s = 0.25;
  EXPECT_EQ(
    go2_cbf_control::fault_response(false, 0.0, grace_s),
    go2_cbf_control::FaultResponse::kHoldLastValidCommand);
  EXPECT_EQ(
    go2_cbf_control::fault_response(false, 0.24, grace_s),
    go2_cbf_control::FaultResponse::kHoldLastValidCommand);
  EXPECT_EQ(
    go2_cbf_control::fault_response(false, 0.25, grace_s),
    go2_cbf_control::FaultResponse::kSlewToStop);
}

TEST(CbfMath, OneHealthyTickImmediatelyCancelsControlledStop) {
  EXPECT_EQ(
    go2_cbf_control::fault_response(true, 100.0, 0.25),
    go2_cbf_control::FaultResponse::kUseCandidate);
}
