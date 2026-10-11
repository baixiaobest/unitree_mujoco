#include <gtest/gtest.h>
#include <random>
#include "go2_cbf_control/cbf_qp.hpp"
using namespace go2_cbf_control;

Config configuration() {
  Config c{}; c.max_lidar_points=64; c.gamma1=c.gamma2=2.;
  c.slack_penalty=1000.; c.max_cbf_slack=.5; c.solver_time_limit_s=.1;
  return c;
}
TEST(WorldQp, RotatedBodyBoundsAndBarriersMatchIndependentBodyProblem) {
  auto config=configuration();
  StaticCbfQp body(config, 1e-7, 10000),world(config, 1e-7, 10000);
  std::mt19937 gen(20261006); std::uniform_real_distribution<double> angles(-3.14,3.14);
  for(int iteration=0;iteration<200;++iteration) {
    const double yaw=angles(gen),c=std::cos(yaw),s=std::sin(yaw);
    std::array<Point,kMaxPoints> pb{},pw{};
    for(int i=0;i<4;++i) {
      double a=angles(gen),r=1.1+.15*i;
      pb[i]={r*std::cos(a),r*std::sin(a),r,false};
      pw[i]={c*pb[i].x-s*pb[i].y,s*pb[i].x+c*pb[i].y,r,false};
    }
    const double nx=.8,ny=-.3,vx=.2,vy=-.1;
    auto b=body.solve(nx,ny,pb,4,vx,vy,.7,-1.3,-.7,1.1,.9);
    auto w=world.solve(c*nx-s*ny,s*nx+c*ny,pw,4,c*vx-s*vy,s*vx+c*vy,.7,-1.3,-.7,1.1,.9,yaw);
    ASSERT_TRUE(b.solved); ASSERT_TRUE(w.solved);
    EXPECT_NEAR(b.u_x,c*w.u_x+s*w.u_y,5e-3);
    EXPECT_NEAR(b.u_y,-s*w.u_x+c*w.u_y,5e-3);
    EXPECT_LE(w.max_constraint_violation,1e-3);
  }
}
TEST(WorldQp, EmptyPointsStillRespectRotatedAsymmetricEnvelope) {
  auto config=configuration(); StaticCbfQp qp(config); std::array<Point,kMaxPoints> p{};
  auto r=qp.solve(5,5,p,0,0,0,.7,-.2,-.4,.3,.6,1.5707963267948966);
  ASSERT_TRUE(r.solved);
  EXPECT_NEAR(r.u_x,.4,1e-3); EXPECT_NEAR(r.u_y,.3,1e-3);
  EXPECT_LE(r.max_constraint_violation,1e-3);
}
TEST(WorldQp, SlackAndInfeasibility) {
  auto config=configuration(); StaticCbfQp qp(config); std::array<Point,kMaxPoints> p{};
  p[0]={.69,0,.69,false};
  auto r=qp.solve(0,0,p,1,0,0,.7,0,0,0,0);
  ASSERT_TRUE(r.solved); EXPECT_GT(r.max_slack,0.); EXPECT_LE(r.max_constraint_violation,1e-3);
  p[0]={.1,0,.1,false};
  r=qp.solve(0,0,p,1,0,0,.7,0,0,0,0);
  EXPECT_FALSE(r.solved);
}
TEST(WorldQp, FiniteTimeLimitedInfeasibleCandidateIsRejected) {
  SolverResult r{}; r.has_candidate=true; r.timed_out_or_iter_limit=true;
  r.max_constraint_violation=.01;
  EXPECT_STREQ(validate_candidate(r,1,.5),"constraint_violation");
  r.max_constraint_violation=0.; EXPECT_EQ(validate_candidate(r,1,.5),nullptr);
  r.slack[0]=.6; EXPECT_STREQ(validate_candidate(r,1,.5),"slack_limit_exceeded");
  r.slack[0]=std::numeric_limits<double>::quiet_NaN();
  EXPECT_STREQ(validate_candidate(r,1,.5),"nonfinite_slack");
}

// Independent analytic projection onto a half-plane with inactive box limits.
TEST(WorldQp, MatchesAnalyticBarrierProjection) {
  auto config=configuration(); StaticCbfQp qp(config,1e-7,10000);
  std::array<Point,kMaxPoints> points{};
  for(int i=0;i<72;++i) {
    double a=i*3.141592653589793/36, x=std::cos(a),y=std::sin(a);
    points[0]={x,y,1,false};
    // At v=.5 toward the point, offset=.5-4+2.04=-1.46.
    // Constraint -2*[x,y].u >= 1.46 => nearest nominal (.2*[x,y]) is -.73*[x,y].
    auto r=qp.solve(.2*x,.2*y,points,1,.5*x,.5*y,.7,-5,-5,5,5,a+.31);
    ASSERT_TRUE(r.solved);
    EXPECT_NEAR(r.u_x,-.73*x,1e-5); EXPECT_NEAR(r.u_y,-.73*y,1e-5);
    EXPECT_LE(r.max_slack,1e-5);
  }
}
