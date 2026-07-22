#!/usr/bin/env bash
# Mapping preflight check -- run AFTER `aloha_bringup use_cameras:=false` and
# `rtabmap_mapping ...` are up, to confirm the IMU-fused mapping pipeline is
# actually healthy before you start driving. Verifies, in dependency order,
# the things that silently failed on 2026-07-14 (IMU orientation, odom dying
# on camera bringup) plus the assumptions added 2026-07-16 that still need a
# live check (IMU QoS, IMU->base_link TF).
#
#   ros2 run is not needed; just:  bash mapping_preflight.sh
#
# Each check prints PASS/FAIL/WARN. Anything FAIL = don't bother driving yet.

# NOTE: no `set -u` -- ROS setup.bash references unset vars and would abort.
source /opt/ros/humble/setup.bash 2>/dev/null
source /home/aloha/interbotix_ws/install/setup.bash 2>/dev/null

pass(){ echo -e "  \033[32mPASS\033[0m  $1"; }
fail(){ echo -e "  \033[31mFAIL\033[0m  $1"; }
warn(){ echo -e "  \033[33mWARN\033[0m  $1"; }
hdr(){  echo; echo "== $1 =="; }

# topic_has_msg <topic> <secs> : true if >=1 msg arrives within <secs>
topic_has_msg(){ timeout "${2:-5}" ros2 topic echo "$1" --once >/dev/null 2>&1; }
# topic_rate <topic> <secs> : prints avg rate or empty
topic_rate(){ timeout "${2:-5}" ros2 topic hz "$1" 2>/dev/null | grep -oE 'average rate: [0-9.]+' | tail -1; }

hdr "1. Hardware"
[ -e /dev/rplidar ] && pass "/dev/rplidar present" || fail "/dev/rplidar missing"
ls /dev/ttyUSB0 >/dev/null 2>&1 && pass "/dev/ttyUSB0 (base serial) present" || warn "/dev/ttyUSB0 missing (base may be on another ttyUSB)"
if command -v rs-enumerate-devices >/dev/null 2>&1; then
  timeout 15 rs-enumerate-devices --compact 2>/dev/null | grep -q 349522070494 \
    && pass "cam_high D435i (349522070494) enumerated" || fail "cam_high D435i NOT enumerated"
fi

hdr "2. Base odometry (must survive camera bringup)"
if topic_has_msg /mobile_base/odom 5; then
  pass "/mobile_base/odom publishing ($(topic_rate /mobile_base/odom 4))"
else
  fail "/mobile_base/odom DEAD -- base serial likely re-enumerated (USB reset)."
fi
if timeout 12 ros2 run tf2_ros tf2_echo odom base_link >/dev/null 2>&1; then
  pass "TF odom -> base_link resolves"
else
  fail "TF odom -> base_link missing (rtabmap can't get poses)"
fi

hdr "3. Camera streams"
for t in /cam_high/camera/color/image_raw /cam_high/camera/aligned_depth_to_color/image_raw /cam_high/camera/color/camera_info; do
  topic_has_msg "$t" 6 && pass "$t" || fail "$t not publishing"
done

hdr "4. IMU fusion (the 2026-07-14 failure)"
if topic_has_msg /cam_high/camera/imu 5; then
  pass "raw /cam_high/camera/imu publishing"
  REL=$(timeout 5 ros2 topic info /cam_high/camera/imu -v 2>/dev/null | grep -A6 Publishers | grep -oiE 'reliability: (reliable|best.?effort)' | head -1)
  echo "        raw IMU QoS -> ${REL:-unknown}  (madgwick handles either)"
else
  fail "raw /cam_high/camera/imu NOT publishing (enable_gyro/accel?)"
fi
# filtered IMU must exist AND carry a non-zero orientation quaternion.
# NB: `ros2 topic echo --once` prints "does not appear to be published yet"
# to STDOUT when there's no publisher, so we can't test for non-empty --
# require an actual 'orientation:' field instead.
FILT=$(timeout 6 ros2 topic echo /cam_high/camera/imu/filtered --once 2>/dev/null | grep -v 'does not appear to be published')
if echo "$FILT" | grep -q 'orientation:'; then
  pass "/cam_high/camera/imu/filtered publishing (madgwick alive)"
  # orientation.w line under 'orientation:'; check any component non-zero
  if echo "$FILT" | grep -A5 '^orientation:' | grep -qE 'w: (0\.[0-9]*[1-9]|[1-9])'; then
    pass "filtered IMU has a real orientation quaternion"
  else
    fail "filtered IMU orientation looks zero/unset"
  fi
  # discover imu frame and verify TF into base_link
  IMU_FRAME=$(echo "$FILT" | grep -m1 'frame_id:' | awk '{print $2}' | tr -d '"')
  if [ -n "$IMU_FRAME" ]; then
    if timeout 12 ros2 run tf2_ros tf2_echo base_link "$IMU_FRAME" >/dev/null 2>&1; then
      pass "TF base_link -> $IMU_FRAME resolves (rtabmap can use the IMU)"
    else
      fail "TF base_link -> $IMU_FRAME MISSING (rtabmap can't place the IMU)"
    fi
  fi
else
  fail "/cam_high/camera/imu/filtered NOT publishing -- madgwick not fusing"
fi

hdr "5. Merged scan + RTAB-Map output"
topic_has_msg /scan 6 && pass "/scan (lidar+depth merged) publishing" || fail "/scan not publishing"
topic_has_msg /rtabmap/map 8 && pass "/rtabmap/map publishing (grid building)" || warn "/rtabmap/map not seen yet (may need a few frames / some motion)"
if timeout 12 ros2 run tf2_ros tf2_echo map base_link >/dev/null 2>&1; then
  pass "TF map -> base_link resolves (rtabmap localized)"
else
  warn "TF map -> base_link not yet (rtabmap still initializing -- often needs first movement)"
fi

echo
echo "Done. Any FAIL above -> fix before driving. Watch the rtabmap terminal:"
echo "  - the '...IMU received doesn't have orientation set...' spam must be GONE"
echo "  - '/mobile_base/odom' must stay alive after the camera comes up"
