# Curl Cheatsheet -- Robot Control API

Copy-paste reference for every route in `docs/VLM_DEMO_API.md` -- read that
doc for what each one actually does, gotchas, and response shapes; this is
just the commands.

Run the export once per terminal session (works from the robot laptop
itself or any other machine on the same LAN -- update the IP if it's
changed, it's DHCP):

```bash
export ROBOT_IP=10.221.1.59
```

## Setup / health

```bash
# Self-describing route directory
curl http://$ROBOT_IP:8080/api

# Live state snapshot (pose, calibrated, nav_status, ...)
curl http://$ROBOT_IP:8080/api/state

# Tell the robot where it is -- REQUIRED once per session before any
# nav goto call will work (arm calls don't need this)
curl -X POST http://$ROBOT_IP:8080/api/set_pose \
  -H 'Content-Type: application/json' \
  -d '{"x": 0, "y": 0, "yaw": 0}'
```

## Locations

```bash
# Every named room/desk + distance from current pose, nearest-first
curl http://$ROBOT_IP:8080/api/locations/all

# Roll-up: nearest room / nearest desk / nearest anything
curl http://$ROBOT_IP:8080/api/nearest

# Floorplan PNG with nearby labels + robot marker (save to a file to view it)
curl http://$ROBOT_IP:8080/api/map_labeled.png -o map.png
```

## Navigation (blocks until arrival/failure/timeout -- L2 must be held on the robot)

```bash
# Go to a named location (fuzzy match: exact/prefix/substring/category word)
curl -X POST http://$ROBOT_IP:8080/api/goto \
  -H 'Content-Type: application/json' \
  -d '{"label": "kitchen", "timeout_s": 180}'

# Go to a raw (x, y)
curl -X POST http://$ROBOT_IP:8080/api/goto_xy \
  -H 'Content-Type: application/json' \
  -d '{"x": 12.3, "y": 4.5, "yaw": 0, "timeout_s": 180}'

# Immediate stop
curl -X POST http://$ROBOT_IP:8080/api/stop
```

## Camera

```bash
# Front camera frame as JPEG (save to a file to view it)
curl "http://$ROBOT_IP:8080/api/camera.jpg?cam=front" -o frame.jpg

# Rear camera (only works if the robot was launched WITHOUT --single-cam)
curl "http://$ROBOT_IP:8080/api/camera.jpg?cam=rear" -o frame_rear.jpg

# Same frame as JSON + base64 (no second file needed)
curl "http://$ROBOT_IP:8080/api/camera?cam=front"
```

## Arm (blocks until motion finishes -- no L2 gate, moves the instant you call it)

```bash
# Wave -- sleep -> wave -> sleep, chained. Takes ~25-30s, use --max-time well above that.
curl --max-time 40 -X POST http://$ROBOT_IP:8080/api/arm/wave \
  -H 'Content-Type: application/json' \
  -d '{"side": "right", "cycles": 3}'

# Extend forward
curl --max-time 15 -X POST http://$ROBOT_IP:8080/api/arm/extend \
  -H 'Content-Type: application/json' \
  -d '{"side": "right"}'

# Open gripper
curl --max-time 15 -X POST http://$ROBOT_IP:8080/api/arm/open_gripper \
  -H 'Content-Type: application/json' \
  -d '{"side": "right"}'

# Close gripper (0.6 = mostly closed, not all the way -- as if holding something)
curl --max-time 15 -X POST http://$ROBOT_IP:8080/api/arm/close_gripper \
  -H 'Content-Type: application/json' \
  -d '{"side": "right", "hold_fraction": 0.6}'

# Retract to TRUE resting/sleep pose (safe to leave it here)
curl --max-time 15 -X POST http://$ROBOT_IP:8080/api/arm/retract \
  -H 'Content-Type: application/json' \
  -d '{"side": "right"}'
```

`side: "left"` will 409 on all of the above until `follower_left`'s
hardware fault is fixed (see `arm_gestures.py`'s header comment) --
`"right"` is the only side that currently works.
