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
