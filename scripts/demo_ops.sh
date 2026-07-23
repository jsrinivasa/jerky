#!/usr/bin/env bash
# Demo-ops helper: bringup -> map -> save -> nav+viewer, with safe teardown.
#
# Why this exists: `ros2 launch` children (robot_state_publisher, joy_node,
# slate_base_node, xs_sdk, etc.) survive killing the launch parent -- learned
# the hard way on 2026-07-14/16. `stop` below kills by executable name, not
# by PID tree, and resets the ros2 daemon so ghost nodes don't linger.
#
# NOTE: no `set -u` -- it aborts when sourcing ROS's setup.bash.
set -e

WS=~/interbotix_ws
PKGDIR="$WS/src/aloha"
RUNDIR=~/.aloha_demo
LOGDIR="$RUNDIR/logs"
mkdir -p "$LOGDIR"

source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"

export DISPLAY="${DISPLAY:-:1}"
export XAUTHORITY="${XAUTHORITY:-/run/user/1000/gdm/Xauthority}"

pidfile() { echo "$RUNDIR/$1.pid"; }

start_bg() {
    # start_bg <name> <cmd...>
    local name="$1"; shift
    nohup "$@" > "$LOGDIR/$name.log" 2>&1 &
    echo $! > "$(pidfile "$name")"
    echo "[$name] started (pid $!), log: $LOGDIR/$name.log"
}

cmd_bringup() {
    local mode="${1:-}"
    # Always tear down first -- found 2026-07-22 that a leftover process
    # from a previous session (specifically nav_joystick_teleop surviving a
    # manual restart that skipped 'stop') is exactly what caused a real
    # choppy-motion bug: two publishers fighting on /nav_cmd_vel with no
    # arbitration. cmd_stop is idempotent (safe on a machine with nothing
    # running yet), so this makes "clean bring-up" automatic instead of
    # relying on remembering a separate 'stop' step every time.
    echo "== Ensuring a clean slate first (stop is idempotent) =="
    cmd_stop
    if [ "$mode" = "nav" ]; then
        echo "== Bringing up base + joy_node + nav-safe teleop (no cameras) =="
        echo "   (use_direct_teleop stays off -- that one fights nav_deadman for L2"
        echo "    on /mobile_base/cmd_vel. use_nav_teleop instead publishes to"
        echo "    /nav_cmd_vel, the same pre-gate topic auto_localize/the planner"
        echo "    use, so nav_deadman still owns the final gated output -- hold L2"
        echo "    and drive with the stick any time nothing else is commanding it.)"
        start_bg bringup ros2 launch aloha aloha_bringup.launch.py \
            use_cameras:=false use_direct_teleop:=false use_nav_teleop:=true
        echo "Run '$0 nav <name>' next -- do NOT drive with the controller until then."
    else
        echo "== Bringing up base + joystick teleop (no cameras) =="
        start_bg bringup ros2 launch aloha aloha_bringup.launch.py use_cameras:=false
        echo "Drive with the controller now. Run '$0 map <name>' when ready to map."
        echo "(Heading straight to nav instead? Use '$0 bringup nav' so teleop_twist_joy"
        echo " doesn't fight nav_deadman for the L2 button.)"
    fi
}

cmd_map() {
    local name="${1:?usage: $0 map <map_name>}"
    echo "== Starting RTAB-Map mapping session: $name =="
    echo "   (IMU fusion + initial_reset:=False fixes are in this launch file)"
    start_bg mapping ros2 launch aloha rtabmap_mapping.launch.py \
        map_name:="$name" rtabmap_viz:=true
    echo "Run '$0 preflight' once it settles (~10-15s), then drive to build the map."
}

cmd_preflight() {
    bash "$PKGDIR/scripts/mapping_preflight.sh"
}

cmd_save() {
    local name="${1:?usage: $0 save <map_name>}"
    mkdir -p ~/maps
    ros2 run nav2_map_server map_saver_cli -f ~/maps/"$name" -t /rtabmap/map
    echo "Saved ~/maps/$name.yaml + .pgm"
}

cmd_maps() {
    echo "== Saved maps (~/maps) =="
    local found=0
    for f in ~/maps/*.yaml; do
        [ -e "$f" ] || continue
        found=1
        local name; name="$(basename "$f" .yaml)"
        printf "  %-30s (saved %s)\n" "$name" "$(date -r "$f" '+%Y-%m-%d %H:%M')"
    done
    [ "$found" = 1 ] || echo "  (none yet -- run '$0 map <name>' then '$0 save <name>')"
}

cmd_localize() {
    echo "== Running auto_localize (rotates in place until 2 visual matches) =="
    echo "   HOLD L2 on the controller now -- this is gated by nav_deadman just"
    echo "   like everything else; it will not move without it."
    ros2 run aloha auto_localize
}

cmd_mapfree() {
    local name="${1:?usage: $0 mapfree <session_name> [--single-cam] [--no-rplidar]}"
    shift || true
    local use_rplidar=true
    local use_cam_low_back=true
    for arg in "$@"; do
        case "$arg" in
            --single-cam)  use_cam_low_back=false ;;
            --no-rplidar)  use_rplidar=false ;;
            *) echo "Unknown flag: $arg (see '$0' with no args for usage)" >&2; exit 1 ;;
        esac
    done
    echo "== Starting map-free navigation: fresh SLAM + live A* + L2 deadman =="
    echo "   (autonomous_mapping.launch.py, use_auto_explore:=false -- goals"
    echo "    come from nav_web_viewer clicks, not frontier auto-exploration)"
    echo "   use_rplidar=$use_rplidar use_cam_low_back=$use_cam_low_back"
    start_bg mapfree ros2 launch aloha autonomous_mapping.launch.py \
        map_name:="$name" rtabmap_viz:=false use_auto_explore:=false \
        use_rplidar:="$use_rplidar" use_cam_low_back:="$use_cam_low_back" \
        use_floorplan_map:=true use_odom_locked_map:=true
    echo "Run '$0 viewer' next -- click an anchor (where the robot is + which"
    echo "way it's facing), Confirm Anchor (this also auto-stops manual nav"
    echo "teleop so it can't fight the planner), then click-to-go as normal."
}

cmd_nav() {
    local name="${1:?usage: $0 nav <map_name> [--no-ekf] [--continuous-mapping]}"
    shift || true
    local use_ekf_odom=true
    local continuous_mapping=false
    for arg in "$@"; do
        case "$arg" in
            --no-ekf) use_ekf_odom=false ;;
            --continuous-mapping) continuous_mapping=true ;;
            *) echo "Unknown flag: $arg (see '$0' with no args for usage)" >&2; exit 1 ;;
        esac
    done
    echo "== Starting navigation (localization + planner + L2 deadman gate) =="
    echo "   Robot moves ONLY while L2 is held on the controller."
    echo "   use_ekf_odom=$use_ekf_odom (wheel+IMU fusion, config/ekf.yaml -- "
    echo "     needs ros-humble-robot-localization installed; --no-ekf rolls"
    echo "     back to raw wheel odom if it's missing or misbehaves)"
    echo "   continuous_mapping=$continuous_mapping (false=locked map, "
    echo "     true=keeps extending the DB live -- see rtabmap_localization.launch.py)"
    start_bg nav ros2 launch aloha navigate_mission.launch.py \
        map_name:="$name" rtabmap_viz:=false \
        use_ekf_odom:="$use_ekf_odom" \
        continuous_mapping:="$continuous_mapping"
    echo "Run '$0 localize' next (hold L2), then '$0 viewer' to pick a destination."
}

cmd_viewer() {
    echo "== Starting nav_web_viewer =="
    start_bg viewer ros2 run aloha nav_web_viewer
    local ip
    ip="$(hostname -I | awk '{print $1}')"
    echo "Open http://${ip}:8080/ -- click a point, Confirm & Go, hold L2 to move, STOP to cancel."
}

cmd_status() {
    echo "== Tracked jobs =="
    for f in "$RUNDIR"/*.pid; do
        [ -e "$f" ] || continue
        local name; name="$(basename "$f" .pid)"
        local pid; pid="$(cat "$f")"
        if kill -0 "$pid" 2>/dev/null; then
            echo "  $name: running (pid $pid)"
        else
            echo "  $name: NOT running (stale pidfile)"
        fi
    done
    echo "== Live ROS-related processes =="
    pgrep -af "ros2|joy_node|slate_base|teleop_node|robot_state_publisher|xs_sdk|rtabmap|realsense|rplidar|nav_deadman|nav_web_viewer|ekf_node|apriltag_node|auto_localize" || echo "  (none)"
}

cmd_stop() {
    echo "== Tearing down: killing by executable name (not PID tree) =="
    pkill -9 -f "ros2 launch|ros2 run aloha nav_web_viewer" 2>/dev/null || true
    pkill -9 -f "robot_state_publisher|teleop_node|joy_node|slate_base_node|xs_sdk" 2>/dev/null || true
    pkill -9 -f "rtabmap|realsense2_camera|rplidar|nav_deadman|simple_nav_planner|nav_web_viewer|laser_scan_merger|depthimage_to_laserscan|imu_filter_madgwick|ekf_node|apriltag_node|auto_localize|static_map_publisher" 2>/dev/null || true
    # base_to_camera_tf / base_to_camera_low_back_tf / base_to_rplidar_tf
    # (static_transform_publisher, part of the nav launch chain) were missing
    # from every pattern above -- found 2026-07-21 that these leak on every
    # 'nav' bringup, same class of bug as the interbotix_gravity_compensation
    # gap below (harmless individually, but 10+ had accumulated since
    # 2026-07-20 and were never killed by 'stop'). base_to_camera_low_back_tf
    # (added same night for the rear camera) was missed by the original fix --
    # "base_to_camera_tf" isn't a substring of "base_to_camera_low_back_tf",
    # so it kept leaking; found again 2026-07-22 with 3 stale generations.
    pkill -9 -f "base_to_camera_tf|base_to_camera_low_back_tf|base_to_rplidar_tf" 2>/dev/null || true
    # interbotix_gravity_compensation (leader_left/leader_right) was missing
    # from every pattern above -- discovered 2026-07-20 that this leaked a
    # pair of these nodes on EVERY bringup all night (20+ pairs accumulated
    # since 18:51, none ever killed by 'stop'), spamming DDS discovery/
    # get_robot_info retries in the background. Prime suspect for that
    # session's mysterious "topics never show up" / USB flakiness symptoms.
    pkill -9 -f "interbotix_gravity_compensation" 2>/dev/null || true
    # {leader,follower}_{left,right}_transform_broadcaster (static_transform_publisher
    # from aloha_bringup.launch.py, world->arm base_link) had the exact same leak --
    # found 2026-07-22 with 21 generations accumulated since 2026-07-20, none ever
    # killed by 'stop' (the "ros2 launch|..." pattern above only kills the launch
    # parent; per the process-management lesson, its static_transform_publisher
    # children survive that and were never covered by any other pattern here).
    pkill -9 -f "_transform_broadcaster" 2>/dev/null || true
    # robot_pose_marker (navigate_mission.launch.py / autonomous_mapping.
    # launch.py) had the exact same leak -- found 2026-07-20 same investigation
    # as gravity_compensation above: 16 copies accumulated since 20:51, one per
    # every 'nav' launch tonight, each sitting at 4-9% CPU forever. Sustained
    # background CPU load from both leaks together is a real suspect for the
    # intermittent camera USB completion errors seen that session (scheduling
    # jitter can starve the kworker threads servicing isochronous USB video).
    pkill -9 -f "robot_pose_marker" 2>/dev/null || true
    sleep 1
    ros2 daemon stop 2>/dev/null || true
    # DDS shared-memory exhaustion: recurred TWICE now (2026-07-21 and again
    # 2026-07-21 late night) after enough rapid restart cycles in one
    # session -- stale /dev/shm/fastrtps_* segments pile up (450+ seen) and
    # new nodes start failing DDS port init ("Failed init_port ...
    # open_and_lock_file failed"), causing bizarre-looking node/topic
    # failures that are easy to misdiagnose as something else entirely (an
    # earlier session burned hours on exactly this before finding it was
    # just this). Clearing it here, every stop, makes the fix automatic
    # instead of relying on remembering it happened before.
    find /dev/shm -maxdepth 1 -name "fastrtps_*" -delete 2>/dev/null || true
    rm -f "$RUNDIR"/*.pid
    echo "== Verify (should be empty) =="
    pgrep -af "ros2|joy_node|slate_base|teleop_node|robot_state_publisher|xs_sdk|rtabmap|realsense|rplidar|gravity_compensation|robot_pose_marker" || echo "  clean."
}

case "${1:-}" in
    bringup)   cmd_bringup "$2" ;;
    map)       cmd_map "$2" ;;
    maps)      cmd_maps ;;
    preflight) cmd_preflight ;;
    save)      cmd_save "$2" ;;
    mapfree)   shift; cmd_mapfree "$@" ;;
    nav)       shift; cmd_nav "$@" ;;
    localize)  cmd_localize ;;
    viewer)    cmd_viewer ;;
    status)    cmd_status ;;
    stop)      cmd_stop ;;
    *)
        cat <<EOF
Usage: $0 <command> [args]

  bringup             Base + joystick teleop only (no cameras). Drive the robot out.
                      Always runs 'stop' first (idempotent) for a clean slate --
                      no need to call 'stop' yourself before bringing up again.
  bringup nav         Base + joy_node only, NO direct teleop (nav mode -- avoids
                      teleop_twist_joy fighting nav_deadman for the L2 button).
  map <name>          Start RTAB-Map mapping (IMU + USB fixes applied). Drive to build map.
  maps                List saved maps in ~/maps (to pick one for 'nav' below).
  preflight           Run mapping_preflight.sh (IMU orientation, TFs, odom, /scan, map).
  save <name>         Save the 2D occupancy grid from the current mapping session.
  mapfree <name> [flags]
                      Map-free session: fresh SLAM + live A* planning + L2 deadman,
                      no prebuilt map needed. Pair with 'viewer' to click an anchor
                      (where the robot is + which way it's facing) then click-to-go.
                      Confirming the anchor auto-stops nav_joystick_teleop so it
                      can't fight the planner on /nav_cmd_vel during the drive.
                      Flags: --single-cam (front camera only, no rear RGBD source)
                             --no-rplidar (camera-derived /scan only, no RPLIDAR)
  nav <name> [flags]  Start navigate_mission (planner + L2 deadman gate) against a saved map.
                      Flags: --no-ekf (raw wheel odom, skip IMU fusion)
                             --continuous-mapping (keep extending the map, don't lock it)
  localize            Run auto_localize (hold L2) -- rotates in place until RTAB-Map
                      confirms 2 visual matches, then reports LOCALIZED.
  viewer              Start nav_web_viewer (click-to-go map page on :8080).
  status              Show tracked jobs + live ROS processes.
  stop                Kill everything (by name, handles orphaned launch children) + reset daemon.

Typical map-free flow (no prebuilt map -- click where the robot is, click where to go):
  $0 bringup nav              # base + joy_node only, no direct teleop
  $0 mapfree session1 --single-cam --no-rplidar  # fresh SLAM session (from scratch);
                              # these flags are what's actually been tested working --
                              # drop them to try the full sensor set instead
  $0 viewer                   # open the URL it prints:
                              #   1. click where the robot physically is right now
                              #   2. click a point it's facing toward, Confirm Anchor
                              #   3. click a destination, Confirm & Go, hold L2
  $0 stop                     # when done

Typical mapping flow:
  $0 bringup                  # drive robot to start position (teleop enabled)
  $0 map building16_imu_v2    # drive to build the map
  $0 preflight                # sanity check mid-session
  $0 save building16_imu_v2   # MUST run before 'stop' -- save reads the live
                              # /rtabmap/map topic, not a file; once rtabmap
                              # is killed there's nothing left to read.
  $0 stop

Typical nav flow (map already exists -- use 'bringup nav', not plain 'bringup'):
  $0 bringup nav              # base + joy_node only, no direct teleop
  $0 maps                     # pick a saved map name
  $0 nav building16_imu_v2
  $0 localize                 # hold L2, watch it confirm localization
  $0 viewer                   # open the URL it prints, click a point, hold L2, repeat
  $0 stop                     # when done
EOF
        exit 1
        ;;
esac
