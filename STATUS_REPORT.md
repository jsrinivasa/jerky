# Mobile ALOHA — Status Report

**Date:** March 20, 2026
**Team:** 2–3 engineers

---

## Big Picture

Our goal is to create a demo with Mobile ALOHA that combines autonomous navigation and manipulation. Specifically: navigate to a location on the Building 16 floor plan and perform a manipulation task (e.g., picking up a banana or pressing an elevator button). We also want to integrate speech-to-text and text-to-speech so we can interact with the robot via voice commands.

---

## Where We Are

### 1. Navigation (Mapping, Localization, Path Planning, Obstacle Avoidance)

We initially explored using ACT (Action Chunking Transformers), a form of imitation learning, for navigation. We found that it doesn't scale well — it requires collecting demonstrations for every new environment and route. We have since moved to standard navigation algorithms, which generalize to any environment given a map.

**a) Mapping — "What does the building look like?"**

We experimented with SLAM (Simultaneous Localization and Mapping), which builds a map by driving the robot through the environment. This works but is time-consuming and impractical for large buildings. We have since moved to using the Building 16 floor plan directly, converting it into an occupancy grid (a grid image where black = walls, white = open space) that the navigation software can use.

We built a conversion tool (`pdf_to_map.py`) that takes a PDF floor plan and produces a ROS-compatible occupancy grid map. We have a working map loaded and viewable in RViz (the robot visualization tool), preserving room labels and text for orientation. The map conversion still needs refinement — the current occupancy grid doesn't perfectly capture all wall boundaries and doorways from the architectural drawing, so path planning sometimes routes through walls. This is a known issue we're actively working on.

**b) Localization — "Where am I on the map?"**

We are using AMCL (Adaptive Monte Carlo Localization) to determine the robot's position on the map. AMCL uses the depth camera to measure distances to nearby walls, then compares those measurements against the map to figure out where the robot is.

We identified and fixed two bugs that were preventing the navigation stack from working:

1. **Wrong camera serial number** — the YAML configuration file had an incorrect serial number for the depth camera, so the system couldn't connect to the camera at all.
2. **Dynamic TF expiration** — the coordinate transform (TF) between the robot base and the camera was being published as a *dynamic* transform, meaning it was broadcast once and then expired after a short time. When the launch file started, it would query this TF, but by the time navigation needed it, the transform had expired and the robot didn't know where the camera was relative to its body. This has been fixed by making the transform static (permanent).
3. **Missing TF link** — a missing `base_footprint → base_link` static transform was breaking the TF chain. The SLATE base driver publishes `odom → base_footprint`, but AMCL and the rest of the nav stack reference `base_link`. Without this link, AMCL couldn't resolve the full transform chain. This has been fixed.

With these fixes, mapping and localization are working significantly better.

We are also researching how to do reliable localization with just one depth camera and no lidar. The plan is to implement an EKF (Extended Kalman Filter) that fuses wheel odometry and visual odometry to provide a robust position estimate. We are consulting with an autonomy professor on proven approaches for this sensor configuration. Since our robot's dynamics differ from standard platforms, any existing EKF implementation will need to be adapted.

**c) Path Planning / Obstacle Avoidance — "How do I get there safely?"**

We have the path planning algorithm implemented using A* (a standard shortest-path algorithm that finds routes around obstacles on the map). We tested it in simulation with the Building 16 floor plan and confirmed that A* correctly finds paths around obstacles. The planner uses a two-stage wall detection approach — identifying wall cores and inflating them by the robot's radius to ensure safe clearance.

Known issue: the path smoothing step (which converts jagged grid paths into smooth curves) currently cuts corners through walls. We have disabled smoothing as a temporary fix and will implement obstacle-aware smoothing next.

### 2. Manipulation

We have run a few experiments with ACT (Action Chunking Transformers) for manipulation tasks. We need to figure out how to integrate either ACT or a VLA (Vision-Language-Action) model to perform simple manipulation tasks in conjunction with navigation.

### 3. Speech-to-Text and Text-to-Speech

The basic code is ready. We will integrate it into the system once the manipulation and navigation pipelines are more defined.

---

## What We Did This Week

1. Fixed the **camera serial number** and **dynamic TF expiration** bugs that were preventing mapping from working.
2. Fixed the **missing `base_footprint → base_link` TF** link that was breaking the navigation stack.
3. Built a **PDF-to-occupancy-grid conversion tool** (`pdf_to_map.py`) to generate navigation maps directly from architectural floor plans.
4. Tested the full **navigation pipeline in simulation** — loaded the Building 16 map, ran A* path planning, and confirmed the planner finds valid routes in RViz.
5. Identified and isolated a **path smoothing bug** where B-spline interpolation cuts through walls; disabled smoothing as a temporary fix.
6. Improved the **inflated map computation** — switched from slow Python loops to OpenCV morphological operations (from ~15s to <0.1s).
7. Created a **3D marble file** of the environment by uploading photos into world models — this will be used to build a realistic simulation environment in Isaac Sim.

---

## Next Steps

1. **Fix map conversion quality** — improve the PDF-to-occupancy-grid conversion so walls, doors, and hallways are accurately represented. The architectural floor plan has double-paned walls, door swing arcs, and text annotations that need to be handled correctly.
2. **Fix obstacle-aware path smoothing** — implement smoothing that respects wall boundaries instead of cutting through them.
3. **Localization on real hardware** — test AMCL with the depth camera on the physical robot now that the TF and camera bugs are fixed.
4. **EKF research** — investigate existing EKF implementations for fusing wheel odometry + visual odometry for localization without lidar.
5. **Isaac Sim environment** — use the marble file to build a realistic 3D simulation of Building 16 in Isaac Sim for testing navigation and manipulation in a virtual environment before deploying on hardware.
6. **End-to-end navigation demo** — navigate the real robot from point A to point B on the Building 16 floor plan.

---
