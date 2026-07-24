#!/usr/bin/env python3
"""
Simple, hand-authored follower-arm gestures for demos: wave, extend/retract,
open/close gripper. Built the same way sleep.py / sleep_right.py command
arms (robot_utils.move_arms/move_grippers -- linear interpolation from
wherever the arm currently is to a target, over a given duration), just
packaged as reusable functions/a class instead of a one-shot CLI script.

Connects to whichever follower arm's OWN xs_sdk driver is already running
(started by aloha_bringup.launch.py, independent of the mobile-base nav
stack) -- this module does not launch or manage that driver itself. Checked
via a bounded wait on that arm's get_robot_info service before touching
anything, so a disconnected/faulted arm fails fast with a clear message
instead of hanging on construction.

Confirmed live 2026-07-23: both leader arms have no serial connection at all
(/dev/ttyDXL_leader_left, _right fail to open) and follower_left's xs_sdk
exited at startup on a motor fault (DYNAMIXEL ID 8 / wrist_rotate not found
-- see ~/.aloha_demo/logs/bringup.log) -- so only follower_right is usable
until that hardware is looked at. wave() still defaults to 'left' per how
it's meant to be used once fixed; it'll raise a clear RuntimeError, not
hang, if called while follower_left is down.

CLI:
    python3 arm_gestures.py wave --side left
    python3 arm_gestures.py wave-from-sleep --side right
    python3 arm_gestures.py extend --side right
    python3 arm_gestures.py retract --side right
    python3 arm_gestures.py open-gripper --side right
    python3 arm_gestures.py close-gripper --side right --hold-fraction 0.6
    python3 arm_gestures.py sleep --side right
    python3 arm_gestures.py wake --side right
"""
from __future__ import annotations

import argparse

from aloha.constants import (
    FOLLOWER_GRIPPER_JOINT_CLOSE,
    FOLLOWER_GRIPPER_JOINT_OPEN,
)
from aloha.robot_utils import (
    get_arm_joint_positions,
    move_arms,
    move_grippers,
    setup_follower_bot,
    sleep_arms,
)
from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,
    robot_shutdown,
    robot_startup,
)
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.srv import RobotInfo

# Matches sleep.py/sleep_right.py's own hardcoded home pose exactly -- the
# "ready" pose the arms are meant to sit in when not doing anything else.
HOME_POSE = [0.0, -0.96, 1.16, 0.0, -0.3, 0.0]
# Arm raised, ready to wave from -- moderate shoulder/elbow lift, well short
# of full extension (see EXTEND_POSE's comment on why that matters).
RAISED_POSE = [0.0, -0.6, 0.6, 0.0, -0.3, 0.0]
# Forward reach for extend(). Deliberately NOT fully straightened (shoulder/
# elbow near 0) -- these arms are mounted close to the mobile base/cameras,
# and a fully outstretched pose from an untested starting configuration is
# exactly the kind of thing worth approaching conservatively first.
EXTEND_POSE = [0.0, -0.3, 0.3, 0.0, 0.0, 0.0]

READY_TIMEOUT_S = 3.0
# How close (radians, per joint) counts as "already there" for skipping a
# redundant raise move in wave().
POSE_TOL = 0.15


class ArmGestures:
    """Minimal follower-arm control for a handful of canned demo gestures."""

    def __init__(self, sides=('left', 'right'), ready_timeout: float = READY_TIMEOUT_S):
        self.node = create_interbotix_global_node('aloha')
        self._bots = {}

        available = []
        for side in sides:
            ns = f'follower_{side}'
            client = self.node.create_client(RobotInfo, f'/{ns}/get_robot_info')
            ready = client.wait_for_service(timeout_sec=ready_timeout)
            self.node.destroy_client(client)
            if ready:
                available.append(side)
            else:
                self.node.get_logger().warn(
                    f"{ns}'s xs_sdk driver isn't up (no /{ns}/get_robot_info "
                    f'service within {ready_timeout:.0f}s) -- skipping. Its '
                    'gestures will raise a clear error if called, not hang. '
                    'Check ~/.aloha_demo/logs/bringup.log for why that '
                    "arm's driver failed to start.")

        for side in available:
            self._bots[side] = InterbotixManipulatorXS(
                robot_model='vx300s', robot_name=f'follower_{side}',
                node=self.node, iterative_update_fk=False)

        robot_startup(self.node)
        # Torques the arm on (position control) -- if it was previously
        # limp/drooped under gravity, expect a small snap into place right
        # here, before any gesture runs. Same setup step sleep.py/
        # sleep_right.py both do first; not specific to this module.
        for bot in self._bots.values():
            setup_follower_bot(bot)

    def _bot(self, side: str):
        if side not in self._bots:
            raise RuntimeError(
                f"follower_{side}'s driver isn't available this session "
                f'(see the warning logged at startup) -- pass sides= to '
                f'include it once the hardware issue is resolved.')
        return self._bots[side]

    def _near(self, bot, pose, tol: float = POSE_TOL) -> bool:
        current = get_arm_joint_positions(bot)
        return all(abs(c - t) <= tol for c, t in zip(current, pose[:5]))

    def wave(self, side: str = 'left', cycles: int = 3, moving_time: float = 0.6):
        """Rock the wrist side to side `cycles` times, then lower back
        home. Raises to RAISED_POSE first, unless the arm is already
        raised or extended (within POSE_TOL) -- in which case it waves
        from right where it is instead of detouring through an extra
        up-then-into-place move."""
        bot = self._bot(side)
        if not (self._near(bot, RAISED_POSE) or self._near(bot, EXTEND_POSE)):
            move_arms([bot], [RAISED_POSE], moving_time=1.5)
        swing = 0.6
        for _ in range(cycles):
            move_arms([bot], [RAISED_POSE[:5] + [swing]], moving_time=moving_time)
            move_arms([bot], [RAISED_POSE[:5] + [-swing]], moving_time=moving_time)
        move_arms([bot], [HOME_POSE], moving_time=1.5)

    def wave_from_sleep(self, side: str = 'right', cycles: int = 3, moving_time: float = 0.6):
        """The demo-ready version of wave(): comes up and waves from
        wherever the arm currently is (see wave()'s own already-raised/
        extended check), then folds down to the true sleep pose
        afterward. Used to also force the arm down to sleep *before*
        waving (guaranteeing a consistent start pose), but that produced
        a visible "turn on, turn off, then wave" stutter that wasn't
        wanted -- removed in favor of just waving from the current pose.
        """
        self.wave(side, cycles=cycles, moving_time=moving_time)
        self.sleep(side)

    def extend(self, side: str = 'right', moving_time: float = 4.0):
        """Slowly extend the arm forward from wherever it currently is."""
        move_arms([self._bot(side)], [EXTEND_POSE], moving_time=moving_time)

    def retract(self, side: str = 'right', moving_time: float = 4.0):
        """Slowly bring the arm back to its rest/home pose."""
        move_arms([self._bot(side)], [HOME_POSE], moving_time=moving_time)

    def open_gripper(self, side: str = 'right', moving_time: float = 2.5):
        move_grippers([self._bot(side)], [FOLLOWER_GRIPPER_JOINT_OPEN], moving_time=moving_time)

    def close_gripper(self, side: str = 'right', moving_time: float = 2.5,
                       hold_fraction: float = 0.6):
        """Close most (not all) the way -- as if holding something.
        hold_fraction=1.0 is fully closed, 0.0 is fully open (default 0.6)."""
        target = FOLLOWER_GRIPPER_JOINT_OPEN - hold_fraction * (
            FOLLOWER_GRIPPER_JOINT_OPEN - FOLLOWER_GRIPPER_JOINT_CLOSE)
        move_grippers([self._bot(side)], [target], moving_time=moving_time)

    def sleep(self, side: str = 'right', moving_time: float = 5.0):
        """Slowly return the arm to its true power-off resting pose -- home
        first, then the arm's own mechanically-stable sleep fold (bot.arm
        .group_info.joint_sleep_positions, from its URDF/config -- the
        exact pose it's designed to rest in under gravity with NO torque,
        which is why nothing further is needed after this to power down
        safely: unlike an arbitrary pose, the sleep fold won't drift or
        droop once you cut power or kill the process, so there's no
        "slam down" to avoid at that point -- this call is where the
        graceful part actually happens, not the powering-off itself.

        This is the exact same helper (robot_utils.sleep_arms) sleep.py/
        sleep_right.py already use -- just callable per-side here instead
        of a separate one-shot script.
        """
        sleep_arms([self._bot(side)], moving_time=moving_time, home_first=True)

    def wake(self, side: str = 'right', moving_time: float = 4.0):
        """Slowly bring the arm back up from wherever it currently is
        (its sleep fold, most likely) to the same ready/home pose extend()
        and retract() use -- the counterpart to sleep(). Functionally
        identical to retract() (both are just move_arms to HOME_POSE); this
        name exists because "wake it back up" is what you're actually
        asking for after a sleep() call, and shouldn't require knowing
        retract() happens to be the same motion.

        setup_follower_bot() (torque + operating modes) already reran when
        this ArmGestures instance was constructed, same as every other
        gesture -- if the arm's torque was actually cut (not just parked in
        its sleep pose with torque still on, which is sleep()'s own end
        state), expect a small snap into place the instant torque
        re-engages, same caveat as __init__'s setup step in general. Once
        that settles, this call proceeds exactly as normal from wherever
        the arm actually is.
        """
        move_arms([self._bot(side)], [HOME_POSE], moving_time=moving_time)

    def shutdown(self):
        robot_shutdown(self.node)


def main(argv=None):
    p = argparse.ArgumentParser(description='Simple follower-arm gesture demos')
    sub = p.add_subparsers(dest='cmd', required=True)

    pw = sub.add_parser('wave')
    pw.add_argument('--side', default='left')
    pw.add_argument('--cycles', type=int, default=3)

    pe = sub.add_parser('extend')
    pe.add_argument('--side', default='right')
    pe.add_argument('--moving-time', type=float, default=4.0)

    pr = sub.add_parser('retract')
    pr.add_argument('--side', default='right')
    pr.add_argument('--moving-time', type=float, default=4.0)

    po = sub.add_parser('open-gripper')
    po.add_argument('--side', default='right')

    pc = sub.add_parser('close-gripper')
    pc.add_argument('--side', default='right')
    pc.add_argument('--hold-fraction', type=float, default=0.6)

    psl = sub.add_parser('sleep')
    psl.add_argument('--side', default='right')
    psl.add_argument('--moving-time', type=float, default=5.0)

    pwk = sub.add_parser('wake')
    pwk.add_argument('--side', default='right')
    pwk.add_argument('--moving-time', type=float, default=4.0)

    pws = sub.add_parser('wave-from-sleep')
    pws.add_argument('--side', default='right')
    pws.add_argument('--cycles', type=int, default=3)

    args = p.parse_args(argv)

    arms = ArmGestures(sides=('left', 'right'))
    try:
        if args.cmd == 'wave':
            arms.wave(args.side, cycles=args.cycles)
        elif args.cmd == 'wave-from-sleep':
            arms.wave_from_sleep(args.side, cycles=args.cycles)
        elif args.cmd == 'extend':
            arms.extend(args.side, moving_time=args.moving_time)
        elif args.cmd == 'retract':
            arms.retract(args.side, moving_time=args.moving_time)
        elif args.cmd == 'open-gripper':
            arms.open_gripper(args.side)
        elif args.cmd == 'close-gripper':
            arms.close_gripper(args.side, hold_fraction=args.hold_fraction)
        elif args.cmd == 'sleep':
            arms.sleep(args.side, moving_time=args.moving_time)
        elif args.cmd == 'wake':
            arms.wake(args.side, moving_time=args.moving_time)
    finally:
        arms.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
