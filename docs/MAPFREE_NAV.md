# Map-Free Click-to-Go Navigation

No prebuilt map needed. Stand at the robot, tell it where it is and which way
it's facing, then click destinations on the floorplan and it drives there.
Built 2026-07-22 to replace needing a pre-saved, pre-localized RTAB-Map
database (see `NAV_RUNBOOK.md` for that older, saved-map flow).

## Operator quick reference (if you're new to this machine)

**Code**: `~/interbotix_ws/src/aloha/` -- the nodes for this flow live in
`aloha/` (`nav_web_viewer.py`, `simple_nav_planner.py`, `nav_deadman.py`),
launch files in `launch/` (`autonomous_mapping.launch.py` is the one this
flow uses), operational tooling in `scripts/` (`demo_ops.sh`), floorplan/room
data in `maps/` (`floorplan.svg`, `floorplan_real_2_nav_rooms_corrected.json`
-- both tracked in git, needed for this to work at all).

**Logs**: `demo_ops.sh` writes each subcommand's full stdout/stderr to
`~/.aloha_demo/logs/<name>.log` (`bringup.log`, `mapfree.log`, `viewer.log`)
-- these are overwritten fresh each time that command runs, not appended, so
"the log" always means the CURRENT/most recent session unless you've copied
it elsewhere first. This directory is NOT in git (correctly -- it's runtime
output) and accumulates across many separate sessions/nights (214MB as of
2026-07-22, going back to 07-16) -- nothing deletes old logs automatically.
Separately, ROS 2's own per-node default logging goes to `~/.ros/log/`
(1.3GB as of 2026-07-22) -- check there if something's missing from the
`demo_ops.sh` logs specifically (e.g. a node that crashed before `demo_ops.sh`
even started capturing its output, or per-node `rosout` detail).

**Saved map-free databases pile up**: each `mapfree <name>` session writes a
live RTAB-Map database to `~/maps/<name>.db` for the duration of that run
(nothing explicitly calls `save` in this flow -- RTAB-Map just always writes
its own db somewhere) -- as of 2026-07-22 there are 20+ `session*.db` files
in `~/maps/` from testing, none of it cleaned up automatically. Harmless to
leave (each session starts fresh via `--delete_db_on_start` regardless of
what's already there), but worth knowing about if `~/maps/` looks cluttered
or disk space matters.

**Status while running**: `demo_ops.sh status` lists tracked jobs + live ROS
processes. The web page's `nav:` HUD badge shows live planner state
(`planning`/`following`/`goal_reached`/`failed (reason)`).

## Quick start

All commands run in the **same terminal**, one after another --
`demo_ops.sh` backgrounds every process it starts (`nohup ... &`) and returns
immediately, so there's no need to open new terminals or windows.

```bash
cd ~/interbotix_ws/src/aloha/scripts

./demo_ops.sh bringup nav                                   # base + joy_node
./demo_ops.sh mapfree <session_name> --single-cam --no-rplidar
./demo_ops.sh viewer                                        # prints the URL
```

Open the printed URL (`http://<robot-ip>:8080/`) in a browser, then:

1. Click where the robot **physically is right now** on the floorplan --
   see "Where exactly is 'the robot is here'?" below, it's not the visual
   center of the footprint.
2. Click a point it's **facing toward** (defines heading).
3. **Confirm Anchor.**
4. Pick a destination either by clicking the floorplan, or by typing a
   room/desk code (e.g. `314`, `D3-12`) into the search box and hitting
   **Go To** on a match -- both stage the same way. **Confirm & Go**, hold
   **L2** on the controller.
5. **STOP** cancels the current drive at any time (see below for what it
   does to manual control).

When done for the day: `./demo_ops.sh stop` tears everything down cleanly.

### Where exactly is "the robot is here"?

Not the front edge, and not the visual center of the footprint --
`base_link` (what the anchor click is actually defining) sits noticeably
**toward the back** of this robot's 28in x 22in base: ~9.3in in front of the
back edge, ~18.7in behind the front edge. Click that point, not the
geometric center.

### Search by room/desk code

The search box matches against `maps/floorplan_real_2_nav_rooms_corrected.json`
(556 labeled rooms/desks, generated from the floorplan SVG's own text labels
by `scripts/svg_to_map.py`'s `extract_rooms_from_svg()`) -- exact match first,
then prefix, then substring, case-insensitive. **Go To** stages the label's
own center/icon coordinate and nothing else -- no separate "stand outside the
door" mode; that was tried (ray-casting for a real doorway) and dropped after
live testing kept landing on clutter or facing a door's swing-arc instead of
the doorway itself. simple_nav_planner's own A* already won't plan through a
wall and already routes to the nearest reachable point if the literal target
can't be reached (see "If the exact goal can't be reached" below) -- that's
a planning-grid concern, not something worth re-solving with floorplan pixel
geometry in the web page.

### `<session_name>`

Free-form label -- becomes the RTAB-Map database filename
(`~/maps/<session_name>.db`). Nothing else depends on it, and each map-free
session starts SLAM from scratch regardless of the name
(`--delete_db_on_start`), so there's no real naming convention to follow.
Anything descriptive works: `office_2026-07-22`, `session14`, etc. -- just
avoid reusing a name you care about keeping, since a later `save` (if you
ever run one) would overwrite that `.db` file.

### `mapfree` flags

- `--single-cam` -- front camera only, no rear RGBD source.
- `--no-rplidar` -- camera-derived `/scan` only, no RPLIDAR S2.

Drop either flag to use the full sensor set. `--single-cam --no-rplidar` is
what's been tested and working as of 2026-07-22.

## How L2 / manual teleop / autonomous driving interact

- **L2 held** is required for the robot to move at all, always (`nav_deadman`
  gate) -- whether you're manually driving with the stick or the planner is
  autonomously driving to a goal.
- **Before you anchor**, `nav_joystick_teleop` is running (started by
  `bringup nav`) -- hold L2 and use the stick to manually drive the robot to
  its actual starting position.
- **Confirming an anchor automatically stops `nav_joystick_teleop`.** This
  is deliberate -- see "Why manual teleop turns off" below. From this point,
  L2 is only satisfying the deadman gate; the stick does nothing.
- **Pressing STOP automatically restarts `nav_joystick_teleop`** so you can
  manually drive again (e.g. back to a starting point after cancelling a bad
  drive), without needing to bring the whole stack down and back up.

## Why manual teleop turns off after anchoring

Found and fixed 2026-07-22. `nav_joystick_teleop` and `simple_nav_planner`
both publish to `/nav_cmd_vel` (the pre-deadman-gate topic). They share the
same L2 enable button as `nav_deadman`. If you hold L2 to satisfy the
deadman while the planner is autonomously driving, `teleop_twist_joy` *also*
sees "enabled" on every joystick message and publishes an all-zero Twist --
at the joystick's native poll rate, much faster than the planner's 10Hz --
onto the same topic, with no arbitration between the two publishers. The
practical symptom was the robot going "smooth for a second, stutter, smooth
again" during autonomous drives: `nav_deadman` was receiving mostly zeros
from teleop, with the planner's real ~0.3 m/s command winning only
occasionally.

This was confirmed, not guessed: `simple_nav_planner`'s own per-tick log
showed a clean, continuous command at steady 10Hz the entire time (the
planner's logic and control-loop timing were never the problem), while
`nav_deadman`'s per-tick log -- what it actually *received* -- showed the
target velocity reading zero on 60-75% of ticks. `ps aux` then confirmed
`nav_joystick_teleop` was still alive and remapped onto `/nav_cmd_vel`.
Killing it made the stutter disappear immediately; reproduced and re-fixed
identically on a second drive to confirm it wasn't a one-off.

The fix (`nav_web_viewer.py`'s `set_anchor()` / `_stop_nav_teleop()`) kills
`nav_joystick_teleop` by its exact node name the moment an anchor is
confirmed -- `joy_node` and `nav_deadman` are untouched, so L2 still gates
all motion exactly as before. The companion fix (`cancel()` /
`_start_nav_teleop()`) relaunches it on STOP so manual control isn't
permanently lost for the rest of the session.

`demo_ops.sh bringup` was also hardened to always run `stop` first
(idempotent, safe even with nothing running) -- so a leftover process from a
previous session can't silently survive into the next one and reintroduce
this same class of bug. This generalizes the project's standing rule: always
do a full clean stop before bringing the stack back up, never a selective
kill of just the node that looks broken.

## A* planning: performance fix + graceful degradation

Both found and fixed 2026-07-22, same investigation. Before these, most
goals were failing outright -- 2 successes out of 17 attempts in one
session -- almost all logged as "A* failed: Maximum iterations (100000)
reached," which reads like "no path exists" but usually wasn't.

**Root cause (performance):** `astar()`'s open-set node selection was
`min(open_set, key=...)` -- a full O(n) linear scan over the entire open
set on *every single iteration*, instead of a real priority queue. On this
floorplan's live grid (padded to ~2.5M cells by `_expand_map_to_include`
for a goal outside the currently-explored area, only ~22% free), the open
set grows large enough that this dominates runtime, and most goals burned
through the full 100,000-iteration budget without ever reaching a target
that was often only a few meters away in a straight line. Fixed by
rewriting `astar()` to use `heapq` (O(log n) per step) -- same A* logic and
cost function, just no longer near-quadratic. Confirmed live: failure rate
dropped from 15/17 to roughly 6/14 immediately after this alone (the
remainder were genuinely expensive/unreachable searches, addressed next).

**If the exact goal still can't be reached:** `astar()` now tracks the
closest-to-goal cell it actually explored (a real, finalized node with a
valid backtrack chain, not a straight-line guess) and, if the literal goal
is never reached within the iteration budget or the open set exhausts
first, returns the route to THAT point instead of failing outright.
`plan_and_navigate`/`attempt_replan_around_obstacle` then sync the
effective goal to wherever the path actually ends, so the robot correctly
recognizes arrival there instead of sitting at a dead end still technically
"short of the goal." This also subsumes the old "goal cell is occupied,
search a 20-cell radius for a free one, fail if none found" pre-check --
that pre-check now falls through to the same best-effort search instead of
failing immediately when its own small local radius comes up empty.

## Turn-specific clearance

If the robot is bumping into things slightly on turns specifically (not a
general "give everything more room" problem): `simple_nav_planner.py`'s
`_widen_turns()`, run on the raw A* path right after planning (before
trajectory smoothing), nudges ONLY waypoints where the path changes
direction sharply (>~29°) outward -- away from the inside of the corner,
along the bisector of the incoming/outgoing travel direction -- by up to
6 inches, verified collision-free against the same inflated map A* used, one
2cm step at a time, stopping at the last safe position. Straight runs are
provably untouched (every point where the path doesn't change direction is
passed through unmodified). `inflation_radius` (A* planned-path clearance
from walls, `autonomous_mapping.launch.py`) stays at the original 0.6048 --
a global bump was tried first and overcorrected, pushing straight-line
segments away from walls too, not just turns, which is a different ask than
what was actually reported.

## What was tried and dropped

Things that were built, tested live, and deliberately abandoned in favor of
what's described above -- kept here so nobody re-discovers the same dead end.

- **"Stand outside the door" goal mode.** Two attempts. First: ray-cast
  doorway detection against `floorplan_occ_px` (the same simple "any dark
  pixel is a wall" mask used for A* planning) -- landed on furniture
  outlines, text remnants, and door swing-arcs as often as real walls, so
  approach points ended up on clutter or facing an arc instead of the
  doorway. Second attempt: switched to `scripts/svg_to_map.py`'s real
  structural-walls-only mask (proper wall classification + morphological
  line extraction that curves can't survive) plus a clearance-disk check
  that pushed the point outward until genuinely verified-free. Both
  individually tested correct in isolation (offline validation against real
  floorplan data), but combined still wasn't reliable enough in live
  testing to trust. Dropped per direct feedback ("lets make it easier") --
  `goto_location` now always targets the room/desk's own center, and
  `simple_nav_planner`'s own A* (which already won't plan through a wall,
  and now degrades to "closest reachable point" instead of failing, see
  above) does the actual "don't end up somewhere bad" job instead. The
  abandoned ray-cast code is `find_room_entrance()` in
  `aloha/navigate_to_room.py` if this is ever worth revisiting with a
  cleaner wall/furniture-classified map as the input.
- **Global `inflation_radius` bump for turn clearance.** Raised from 0.6048
  to 0.681 (+3in), then 0.643 (+1.5in) after live feedback that 0.681 was
  too much. Both applied clearance to EVERY segment of every path, not just
  turns -- overcorrected straight-line corridors (routes visibly hugging
  corridor centers). Reverted to the original 0.6048; replaced with
  `_widen_turns()`, which only touches waypoints where the path actually
  changes direction.
- **Killing `nav_joystick_teleop` unconditionally at anchor-confirm, with no
  way back.** First version of the teleop-vs-planner fix did this and
  nothing else -- worked for the stutter bug, but then STOP had no way to
  restore manual control, leaving no way to drive the robot back to a
  start position after cancelling a bad drive short of a full stack
  restart. Added `_start_nav_teleop()` as the companion, triggered by STOP,
  so manual control comes back without losing the current anchor/SLAM
  session.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Motion is choppy/stuttery during an autonomous drive | Something else is also publishing to `/nav_cmd_vel` | `ps aux \| grep nav_joystick_teleop` -- if it's running mid-drive, that's almost certainly it. Should not happen anymore after the 2026-07-22 fix unless the anchor step was bypassed. |
| Stick does nothing after anchoring | Expected -- see above. Press STOP to get it back, or re-run `bringup nav` for a full restart. |
| `/floorplan_map` never appears / planner errors "not anchored" | No anchor confirmed yet this session | Click anchor position + heading + Confirm Anchor in the viewer first. |
| Robot moves the instant L2 is released | Should not happen -- `nav_deadman` ramps to zero over ~0.25s on release, doesn't cut instantly | Physical E-stop is the true instant kill if this is ever observed. |
| A goal fails to plan / "Maximum iterations reached" in the log | Should be rare now (see A* section above) -- if it still happens, the closest-approach fallback should have kicked in instead of a hard failure; check the `nav:` status badge on the page and `/nav_status` in the log for the actual outcome. |
| A second goal shows the FIRST goal's route instead of a new one | Should not happen anymore -- `/smoothed_path` used to be left stale on a failed replan (TRANSIENT_LOCAL topic, never cleared). Fixed by explicitly publishing an empty path on failure. Check the `nav:` status badge -- it now shows `failed (reason)` in red instead of silently doing nothing. |
