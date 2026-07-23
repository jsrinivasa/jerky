#!/usr/bin/env python3
"""
Nav Deadman Gate ("hold-to-run" safety override)

Sits between the planner and the base. It republishes the planner's velocity
to the base's cmd_vel topic ONLY while a controller enable button (L2) is held.
Any other condition -> it ramps to zero velocity over a short bounded window
(~0.25s, same anti-tip rationale as simple_nav_planner.stop_robot's ramp),
actively holding the robot stopped once there. So "not holding the button" is
the safe default, not something the operator has to react to.

It ramps toward zero when ANY of these is true:
  - the enable button is not currently held,
  - no joystick message has arrived recently (controller off/disconnected),
  - the planner hasn't published a command recently (planner died/idle).

Output is slew-rate limited (accel-capped) in BOTH directions, not just a
passthrough-or-zero switch:
  - On engage, the planner's target may already be well above zero -- it
    keeps computing its own internally-ramped velocity even while gated off,
    so the first command available the instant L2 is pressed can be near
    max speed. Ramping the deadman's own output from 0 up to that target
    (instead of passing it through raw) is what actually makes the robot
    accelerate smoothly from a stop.
  - On disengage, the output ramps down over the same bounded window instead
    of a single instant zero, to avoid a hard jolt/tip. This is a deliberate
    tradeoff: the robot travels a small bounded extra distance (roughly
    0.5 * v * ramp_time -- a few cm at typical speeds) after L2 is released,
    in exchange for not slamming to a stop. The base's physical E-STOP below
    remains the true instant kill.

Wiring (see navigate_mission.launch.py):
    simple_nav_planner  /cmd_vel -> /nav_cmd_vel   (planner output, gated)
    nav_deadman         /nav_cmd_vel -> /mobile_base/cmd_vel  (only while L2 held)
    joy_node            /mobile_base/joy            (button source)
The joystick TELEOP node must NOT also be publishing /mobile_base/cmd_vel, or
it will fight this gate -- run joy_node only (no teleop_twist_joy) in nav mode.

This is the operational stop. The base's PHYSICAL E-STOP remains the top-level
hardware kill (cuts motor power, no ramp, no software in the loop) and is
independent of all of this.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy,
)

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy


class NavDeadman(Node):

    def __init__(self):
        super().__init__('nav_deadman')

        self.declare_parameter('enable_button', 6)          # L2 (button index 6)
        self.declare_parameter('joy_topic', '/mobile_base/joy')
        self.declare_parameter('input_topic', '/nav_cmd_vel')
        self.declare_parameter('output_topic', '/mobile_base/cmd_vel')
        self.declare_parameter('rate_hz', 20.0)
        self.declare_parameter('joy_timeout', 0.5)          # s; stale joy -> stop
        self.declare_parameter('cmd_timeout', 0.5)          # s; stale planner cmd -> stop
        # Slew-rate limits on the OUTPUT (not the planner's own ramp -- see
        # module docstring). Defaults give ~0.25s for a full 0<->max ramp at
        # navigate_mission.launch.py's default max_linear/angular_velocity
        # (0.5 m/s, 1.5 rad/s). Applied identically on engage and disengage.
        self.declare_parameter('linear_accel_limit', 2.0)    # m/s^2
        self.declare_parameter('angular_accel_limit', 6.0)   # rad/s^2

        self.enable_button = int(self.get_parameter('enable_button').value)
        joy_topic = self.get_parameter('joy_topic').value
        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        rate_hz = float(self.get_parameter('rate_hz').value)
        self.joy_timeout = float(self.get_parameter('joy_timeout').value)
        self.cmd_timeout = float(self.get_parameter('cmd_timeout').value)
        self.linear_accel_limit = float(self.get_parameter('linear_accel_limit').value)
        self.angular_accel_limit = float(self.get_parameter('angular_accel_limit').value)
        self._dt = 1.0 / rate_hz

        self._last_cmd = Twist()
        self._last_cmd_time = None
        self._last_joy_time = None
        self._enabled = False
        self._was_enabled = None  # for logging transitions
        self._out_linear_x = 0.0
        self._out_angular_z = 0.0

        # Joy: subscribe BEST_EFFORT so we receive it whether the publisher is
        # reliable or best-effort (avoids a silent QoS-mismatch drop of the
        # very signal that gates motion).
        joy_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        self.create_subscription(Joy, joy_topic, self._joy_cb, joy_qos)
        self.create_subscription(Twist, input_topic, self._cmd_cb, 10)
        self._pub = self.create_publisher(Twist, output_topic, 10)

        self.create_timer(1.0 / rate_hz, self._tick)

        self.get_logger().warn(
            f'Deadman ACTIVE: robot moves ONLY while button {self.enable_button} '
            f'(L2) is held on {joy_topic}. Engage/release both ramp '
            f'(accel-limited: {self.linear_accel_limit} m/s^2 linear, '
            f'{self.angular_accel_limit} rad/s^2 angular), release is NOT instant. '
            f'Gating {input_topic} -> {output_topic}.')

    def _joy_cb(self, msg: Joy):
        self._last_joy_time = self.get_clock().now()
        held = (len(msg.buttons) > self.enable_button
                and msg.buttons[self.enable_button] == 1)
        self._enabled = held

    def _cmd_cb(self, msg: Twist):
        self._last_cmd = msg
        self._last_cmd_time = self.get_clock().now()

    def _fresh(self, t, timeout):
        if t is None:
            return False
        return (self.get_clock().now() - t).nanoseconds / 1e9 <= timeout

    @staticmethod
    def _slew(current: float, target: float, max_delta: float) -> float:
        if target > current:
            return min(target, current + max_delta)
        return max(target, current - max_delta)

    def _tick(self):
        joy_ok = self._fresh(self._last_joy_time, self.joy_timeout)
        cmd_ok = self._fresh(self._last_cmd_time, self.cmd_timeout)
        allow = self._enabled and joy_ok and cmd_ok

        target = self._last_cmd if allow else Twist()  # Twist() == all-zero
        self._out_linear_x = self._slew(
            self._out_linear_x, target.linear.x,
            self.linear_accel_limit * self._dt)
        self._out_angular_z = self._slew(
            self._out_angular_z, target.angular.z,
            self.angular_accel_limit * self._dt)

        out = Twist()
        out.linear.x = self._out_linear_x
        out.angular.z = self._out_angular_z
        # Untrottled per-tick log (20Hz) -- pairs with simple_nav_planner's
        # own [control-tick] log so a choppy-motion diagnosis can compare
        # what the PLANNER computed vs. what ACTUALLY reached the base
        # after slew-limiting/gating here. Added for a short, controlled
        # test drive -- not meant to run unthrottled long-term.
        self.get_logger().info(
            f'[deadman-tick] allow={allow} joy_ok={joy_ok} cmd_ok={cmd_ok} '
            f'target_lin={target.linear.x:.3f} target_ang={target.angular.z:.3f} '
            f'out_lin={out.linear.x:.3f} out_ang={out.angular.z:.3f}'
        )
        self._pub.publish(out)

        if allow != self._was_enabled:
            if allow:
                self.get_logger().info('L2 held -> ramping planner velocity to base')
            else:
                reason = ('button released' if not self._enabled
                          else 'joystick signal lost' if not joy_ok
                          else 'planner idle/stale')
                self.get_logger().warn(f'STOP ({reason}) -> ramping down to zero velocity')
            self._was_enabled = allow


def main(args=None):
    rclpy.init(args=args)
    node = NavDeadman()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # best-effort: leave a zero on the wire on the way out
        try:
            node._pub.publish(Twist())
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
