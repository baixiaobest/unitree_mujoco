"""Relative odometry integration from sensor-timed body-frame velocities."""

import math


def quat_multiply(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    )


def rotate(q, v):
    p = quat_multiply(quat_multiply(q, (0.0, *v)), (q[0], -q[1], -q[2], -q[3]))
    return p[1:]


class LearnedMotionIntegrator:
    def __init__(self):
        self.position = (0.0, 0.0, 0.0)
        self.orientation = (1.0, 0.0, 0.0, 0.0)
        self.last_stamp_ns = None
        self.last_sequence = None
        self.last_linear = None
        self.last_angular = None

    def reset_time_reference(self):
        self.last_stamp_ns = None
        self.last_sequence = None
        self.last_linear = None
        self.last_angular = None

    def update(self, stamp_ns, sequence, linear, angular):
        if self.last_sequence is not None and sequence <= self.last_sequence:
            return False
        if not all(math.isfinite(x) for x in (*linear, *angular)):
            return False
        if self.last_stamp_ns is not None:
            dt = (stamp_ns - self.last_stamp_ns) * 1e-9
            if dt < 0.0:
                return False
            if 0.0 < dt <= 0.2 and self.last_linear is not None:
                # The previous estimate describes the interval until this sample.
                omega = self.last_angular
                speed = math.sqrt(sum(x*x for x in omega))
                quarter_angle = speed * dt * 0.25
                if speed > 1e-9:
                    scale = math.sin(quarter_angle) / speed
                    dq_half = (math.cos(quarter_angle), *(scale*x for x in omega))
                else:
                    dq_half = (1.0, *(0.25*dt*x for x in omega))
                q_mid = quat_multiply(self.orientation, dq_half)
                v_world = rotate(q_mid, self.last_linear)
                self.position = tuple(p + dt*v for p, v in zip(self.position, v_world))
                self.orientation = quat_multiply(q_mid, dq_half)
                norm = math.sqrt(sum(x*x for x in self.orientation))
                self.orientation = tuple(x/norm for x in self.orientation)
        self.last_stamp_ns = stamp_ns
        self.last_sequence = sequence
        self.last_linear = linear
        self.last_angular = angular
        return True
