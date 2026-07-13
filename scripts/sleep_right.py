#!/usr/bin/env python3
"""Sleep just the right follower arm (no leader dependencies)."""
import time
import numpy as np
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS

HOME_POSE = [0.0, -0.96, 1.16, 0.0, -0.3, 0.0]


def move_smooth(bot, target, moving_time=3.0, dt=0.05):
    """Smooth interpolation from current to target."""
    num_steps = int(moving_time / dt)
    current = np.array(bot.arm.get_joint_commands())
    target = np.array(target)
    for t in range(num_steps):
        alpha = (t + 1) / num_steps
        pos = current + alpha * (target - current)
        bot.arm.set_joint_positions(list(pos), moving_time=dt*2, blocking=False)
        time.sleep(dt)


def main():
    bot = InterbotixManipulatorXS(
        robot_model='vx300s',
        robot_name='follower_right',
    )

    print(">>> Moving to home pose (staged)...", flush=True)
    move_smooth(bot, HOME_POSE, moving_time=3.0)
    time.sleep(0.5)

    print(">>> Moving to sleep pose...", flush=True)
    sleep_pose = list(bot.arm.group_info.joint_sleep_positions)
    print(f"    sleep joint angles: {sleep_pose}", flush=True)
    move_smooth(bot, sleep_pose, moving_time=3.0)
    time.sleep(0.5)

    print(">>> Done. Arm is in sleep pose.", flush=True)


if __name__ == '__main__':
    main()
