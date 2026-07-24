# Robot Control API for a VLM / Speech Demo

For an agent building a speech-to-speech demo (on a **different laptop**)
that lets a user talk to the robot -- ask what it sees, ask where it is,
tell it to go somewhere -- and have a live conversation while it drives.

The robot's onboard laptop already exposes an HTTP API for exactly this.
You do not need ROS, rclpy, or anything installed on your laptop -- it's
plain HTTP/JSON, callable from Python (`requests`), a browser, or `curl`.

## 0. Before you write any code

A human needs to have the nav stack up on the **robot's** laptop and be
**physically holding L2** on the Xbox controller the whole time the robot is
expected to move (a hardware kill-switch, independent of this API -- see
"Gotchas" below). That's their side, not yours; just know that if a `/api/goto`
call times out with no motion, check that first before assuming your code is
broken.

Confirm reachability from your laptop before writing anything else:
```bash
curl http://<robot-ip>:8080/api
```
`<robot-ip>` is the robot laptop's current LAN IP (get it from whoever's
running the robot -- it's DHCP-assigned and can change between sessions).
Both laptops must be on the same LAN/Wi-Fi. There's no auth and no HTTPS on
this server -- it's meant for a trusted local network only, not the internet.

## 1. API reference

All bodies are JSON. All responses are JSON except the two `.jpg` routes.
Units: meters, radians, seconds throughout.

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/api` | -- | Self-describing directory of every route below |
| POST | `/api/set_pose` | `{x, y, yaw}` | `{ok, transform, message}` -- tells the robot where it is. **Must succeed once before any goto call will work.** A human normally does this once at the start of a session (it's the same "click where I am + which way I'm facing" step as the web UI) -- your script probably won't call this itself unless you're also driving the whole session. |
| GET | `/api/locations/all` | -- | List of every named room/desk: `{label, category, x, y, distance_m}`, **sorted nearest-first**. `distance_m` is `null` if `set_pose` hasn't succeeded yet. |
| GET | `/api/nearest` | -- | Convenience roll-up for "where are you" queries: `{room, desk, any, calibrated}` -- each of `room`/`desk`/`any` is a single nearest-match object (same shape as one `locations/all` entry) or `null`. Use this instead of re-deriving room-vs-desk filtering yourself (see below for why that's not obvious from category alone). |
| GET | `/api/map_labeled.png?radius=12&max_labels=40` | -- | Floorplan PNG with nearby labels + a marker/heading-arrow for the robot's own position, cropped to `radius` meters around it (`radius=full` for the whole building). For when you want the VLM to reason **visually** about "where am I" from a picture of a map, rather than off the numeric `/api/nearest` JSON. |
| POST | `/api/goto` | `{label, timeout_s=180}` | Resolves `label` (fuzzy: exact -> prefix -> substring against all labels, **plus** a category-word fallback -- `"elevator"`, `"bathroom"`, `"conference"`, etc. -- see below; nearest match wins any tie) and drives there. **Blocks until arrival, failure, or timeout** -- the HTTP response *is* the "done" notification: `{ok, label, category, distance_m, status, elapsed_s}`. `status` is `"goal_reached"`, `"failed:<reason>"`, `"idle"` (stopped via `/api/stop` mid-drive), or `"timeout"`. |
| POST | `/api/goto_xy` | `{x, y, yaw=0, timeout_s=180}` | Same blocking contract as `/api/goto`, but to a raw coordinate instead of a named label. |
| GET | `/api/camera.jpg?cam=front\|rear` | -- | Latest camera frame as a raw JPEG. 204 (empty) if that camera has no frame yet. This is your answer to "what do you see" -- feed the bytes straight to your VLM. |
| GET | `/api/camera?cam=front\|rear` | -- | Same frame as JSON: `{ok, cam, width, height, mimetype, image_base64}`, if you'd rather not do a second HTTP round-trip for image bytes. |
| GET | `/api/state` | -- | Live snapshot: `{pose:{x,y,yaw}, calibrated, nav_status, staged_goal, path, ...}`. `nav_status` is `"idle"`, `"planning"`, `"following"`, `"goal_reached"`, or `"failed:<reason>"` -- poll this if you want to narrate progress *during* a drive (see pattern below). |
| POST | `/api/stop` | -- | Immediate stop -- cancels whatever's in flight (staged goal, active drive) and publishes `/nav_cancel`, which the planner treats as a hard kill switch regardless of what it's doing. Call this for a "stop"/"stop moving" intent. If a `/api/goto` call is blocking in a background thread when you call this, that thread's response will come back shortly after with `status: "idle"` (`ok: false`) -- it unblocks within about a poll interval, it doesn't hang until `timeout_s`. (`/api/cancel` is the same action under its original name, kept for the human web page's button -- use `/api/stop`, it's the one meant for this.) |

## 1a. Arm control reference

A **separate hardware subsystem** from everything above -- not gated by
`set_pose`/anchoring at all, and not gated by the physical L2 kill-switch
either (that's the mobile base's safety interlock specifically; the arm has
no equivalent hardware gate, so an arm call moves the arm the moment you
send it, full stop). All block until the motion finishes and return
`{ok, side, message}` -- same "the HTTP response is the completion
notification" contract as `/api/goto`.

| Method | Path | Body | What it does |
|---|---|---|---|
| POST | `/api/arm/wave` | `{side="right", cycles=3}` | Raises the arm (unless it's already raised/extended) and rocks the wrist `cycles` times, then folds down to the resting/sleep pose. **Takes ~15-20s** -- use a generous client-side timeout (30s+). |
| POST | `/api/arm/extend` | `{side="right", moving_time=4.0}` | Slowly reaches forward from wherever the arm currently is. ~4s. |
| POST | `/api/arm/open_gripper` | `{side="right", moving_time=2.5}` | Slowly opens the gripper fully. ~2.5s. |
| POST | `/api/arm/close_gripper` | `{side="right", moving_time=2.5, hold_fraction=0.6}` | Slowly closes the gripper. `hold_fraction` 0-1, default 0.6 = mostly closed but not all the way (as if holding something); 1.0 = fully closed. ~2.5s. |
| POST | `/api/arm/retract` | `{side="right", moving_time=5.0}` | Slowly returns to the arm's TRUE resting/sleep pose (not a mid-way "ready" pose) -- safe to leave the arm there indefinitely afterward. ~5-10s (home first, then the sleep fold). |

**`side` defaults to `"right"` and, as of this writing, `"right"` is the
ONLY side that actually works** -- `"left"` will return a `409` with a clear
message (`follower_left`'s driver has a hardware fault: a motor isn't being
detected at startup). Don't build UI/voice handling that assumes both arms
respond; if you want to support "wave your left arm" as a spoken command,
have it fail gracefully with whatever message comes back in `message`.

**These 5 moves are the entire current repertoire** -- not a hardware
limit, just what's been wired up as of this writing. The arm is a full
6-DOF manipulator (waist/shoulder/elbow/forearm_roll/wrist_angle/
wrist_rotate) and can physically reach far more than these five fixed
poses; there's just no endpoint yet for an arbitrary target. If your demo
needs the arm to reach toward something specific (not just one of these
five canned spots), that's a real gap to flag back rather than something
to work around client-side.

### Room vs. desk labels

`locations/all` categories are: `numbered_room` (plain numbers, e.g. `314`,
`301`), `conference_room`, `corridor`, `restroom`, `vertical_transport`
(elevators/stairs), `utility`, and `named_space` (a catch-all that includes
break rooms, area labels like `A`, `B`, and -- specifically -- **every desk**,
labeled like `D3-12`). There's no separate `desk` category. `/api/nearest`
already applies this so you don't have to, but if you're filtering
`locations/all` yourself for some other reason:
- **desk**: filter labels matching `^D\d` (e.g. `D3-12`)
- **room**: category in `{numbered_room, conference_room}`

### Is there an API for "what's on the desk in front of you"?

No, and it's not a gap -- that's an **object-level visual question about
the current frame**, which is exactly what your VLM is for, not something
the robot's API layer tries to answer itself (it has no VLM/object
detection of its own). Fetch `/api/camera.jpg` and ask your own VLM. The
robot API's job stops at handing you the raw sensor/position data
(camera frame, pose, distances); interpreting it is the demo script's job.

## 2. Gotchas

- **`/api/goto` and `/api/goto_xy` block for the whole drive** (up to
  `timeout_s`). If you want the demo to keep talking *while the robot
  moves* (the "have conversations as it's moving" ask), don't just call it
  synchronously in your main loop -- run it in a background thread and poll
  `/api/state`'s `nav_status` (or just chat freely and check the thread) for
  progress. Pattern below.
- **Nothing moves without L2 held on the robot end**, API call or not. A
  `/api/goto` that never gets physical L2 will just run out its `timeout_s`
  and return `status: "timeout"` -- that's not a bug in your client.
  Handle it as an ordinary user-facing failure ("looks like I can't go
  right now").
- **`set_pose` gates everything.** If `distance_m` is always `null` and
  `goto` always fails with "not anchored", it means nobody's called
  `set_pose` (or the equivalent anchor click on the web page) yet this
  session -- that's a one-time step per session, not per-request.
- **Camera availability depends on how the robot was launched** -- if it was
  brought up with `--single-cam`, only `cam=front` will ever return a frame;
  `cam=rear` will 204 forever. Ask the robot operator which flags were used
  if `rear` isn't working and you expect it to be.
- **IP can change.** It's DHCP, not a fixed address -- if calls start
  timing out that previously worked, re-confirm the IP before debugging
  anything else.
- **Arm calls have no L2/kill-switch gate.** Unlike nav, there's no
  physical hold-to-move interlock for the arm -- a `/api/arm/*` call moves
  it immediately. Nothing to check before it'll work, but also nothing
  stopping it besides `/api/arm/*` not being called.
- **`/api/arm/wave` takes ~15-20s** -- it's wave -> sleep-pose chained as
  one call, not a quick gesture. Use a client timeout well above that
  (30s+), and don't assume a call that's taking 10+ seconds has hung.

## 3. Suggested mapping to demo scenarios

**"What are you seeing right now?"**
```python
import requests
frame = requests.get(f"http://{ROBOT_IP}:8080/api/camera.jpg", params={"cam": "front"}).content
# frame is JPEG bytes -- pass directly to your VLM's image input
```

**"Where are you? / What room are you closest to? / What desk are you closest to?"**
```python
nearest = requests.get(f"http://{ROBOT_IP}:8080/api/nearest").json()
# nearest = {"room": {...} or None, "desk": {...} or None, "any": {...}, "calibrated": bool}
closest_room, closest_desk = nearest["room"], nearest["desk"]
```
If you'd rather have the VLM answer visually (e.g. it's easier for your
speech loop to just *describe a picture* than reason over JSON), pull an
annotated map crop instead:
```python
map_png = requests.get(
    f"http://{ROBOT_IP}:8080/api/map_labeled.png", params={"radius": 12}
).content  # PNG bytes: nearby labels + a marker for the robot's own position
```

**"Go to <place>" -- while staying conversational during the drive**
```python
import threading, requests

def _drive(label, result_holder):
    result_holder["result"] = requests.post(
        f"http://{ROBOT_IP}:8080/api/goto",
        json={"label": label, "timeout_s": 180},
    ).json()

result_holder = {}
t = threading.Thread(target=_drive, args=("jaws", result_holder), daemon=True)
t.start()

# main loop: keep listening/talking to the user here while t runs.
# Poll for progress/completion without blocking your conversation loop:
while t.is_alive():
    status = requests.get(f"http://{ROBOT_IP}:8080/api/state").json()["nav_status"]
    # e.g. narrate "still on my way" / "almost there" based on `status`
    ...  # your speech loop tick goes here

print(result_holder["result"])  # {"ok": True/False, "status": "goal_reached"/"failed:..."/"idle"/"timeout", ...}
```

This is the one pattern worth internalizing: **`goto`/`goto_xy` are the
"done" signal by virtue of blocking**, so the natural way to stay
conversational during a drive is a background thread (or async task) plus
polling `/api/state`, not switching to a non-blocking API -- there isn't
one, by design, since a blocking call is exactly what a synchronous
tool-call-style integration wants.

**"Wave hello" / "pick something up" (canned arm gestures)**
```python
import requests
requests.post(f"http://{ROBOT_IP}:8080/api/arm/wave", json={}, timeout=40)
requests.post(f"http://{ROBOT_IP}:8080/api/arm/extend", json={}, timeout=15)
requests.post(f"http://{ROBOT_IP}:8080/api/arm/close_gripper", json={"hold_fraction": 0.6}, timeout=15)
# ... whatever the demo does while "holding" something ...
requests.post(f"http://{ROBOT_IP}:8080/api/arm/open_gripper", json={}, timeout=15)
requests.post(f"http://{ROBOT_IP}:8080/api/arm/retract", json={}, timeout=15)
```
Same blocking contract as nav -- these don't need the background-thread
pattern above unless you specifically want to talk *during* an arm move
too (they're short enough, except `wave`, that it's usually fine to just
wait).
