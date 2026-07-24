#!/usr/bin/env bash
# Interactive, one-step-at-a-time tester for aloha/arm_gestures.py.
#
# Each gesture runs as its own foreground `ros2 run` process -- Ctrl-C at
# ANY point (during a prompt or mid-motion) kills the script and whatever
# step is running right then, immediately. move_arms/move_grippers send one
# position command per 20ms tick in a plain loop (see robot_utils.py), so a
# kill just stops sending new commands -- the arm holds wherever it was,
# nothing keeps running in the background.
#
# Requires the arm drivers already up (whatever brought follower_right's
# xs_sdk up this session -- aloha_bringup.launch.py). This script only
# ros2 run's the gesture CLI; it doesn't launch or manage the drivers
# themselves. If a step's driver isn't available, arm_gestures.py fails
# that step fast with a clear message instead of hanging (confirmed
# 2026-07-23: follower_left is currently down on a motor fault -- see
# arm_gestures.py's own header comment -- so this script only exercises
# follower_right).
set -eo pipefail

cd ~/interbotix_ws
# colcon's generated setup.bash references some env vars (e.g.
# COLCON_TRACE) without guarding them, so it isn't nounset-safe -- source
# it before turning -u on, not after.
source install/setup.bash
set -u

run_step() {
    local desc="$1"; shift
    echo
    echo "=== ${desc} ==="
    echo "    ros2 run aloha arm_gestures $*"
    read -rp "    Press Enter to run this step (Ctrl-C to abort here instead): " _
    ros2 run aloha arm_gestures "$@"
    echo "    done."
}

echo "Arm gesture tester -- one step at a time, Enter to advance, Ctrl-C to abort."
read -rp "Confirm the area around the RIGHT arm is clear, then press Enter to begin: " _

run_step "1/6  Open right gripper"                          open-gripper --side right
run_step "2/6  Close right gripper (60% -- not all the way)" close-gripper --side right --hold-fraction 0.6
run_step "3/6  Open right gripper again"                    open-gripper --side right
run_step "4/6  Retract right arm to home"                   retract --side right
run_step "5/6  Extend right arm forward"                    extend --side right
run_step "6/6  Retract right arm to home"                   retract --side right

echo
echo "All steps done."
echo
echo "Note: wave defaults to --side left, which will fail fast right now"
echo "(follower_left's driver is down -- a real hardware fault, not this"
echo "script). To sanity-check the wave motion itself today, run it"
echo "manually against the arm that's actually working:"
echo "    ros2 run aloha arm_gestures wave --side right"
echo "then switch back to --side left once that hardware issue is fixed."
