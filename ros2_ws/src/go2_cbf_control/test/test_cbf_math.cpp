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

TEST(CbfMath, YawGovernorLimitsRateAndAcceleration) {
  // At 50 Hz and 1 rad/s^2, a target may alter the command by 0.02 rad/s.
  EXPECT_DOUBLE_EQ(go2_cbf_control::limited_yaw_command(2.0, 0.0, 0.6, 1.0, 0.02), 0.02);
  EXPECT_DOUBLE_EQ(go2_cbf_control::limited_yaw_command(-2.0, 0.6, 0.6, 1.0, 0.02), 0.58);
  EXPECT_DOUBLE_EQ(go2_cbf_control::limited_yaw_command(2.0, 2.0, 0.6, 1.0, 0.02), 0.6);
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
