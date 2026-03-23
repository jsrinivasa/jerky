#!/usr/bin/env python3
"""
Automated sensor integration test for ALOHA system.

Run with bringup already launched in another terminal:
    ros2 launch aloha aloha_bringup.launch.py use_gravity_compensation:=false

Usage:
    python3 ~/interbotix_ws/src/aloha/scripts/test_sensor_integration.py
    python3 ~/interbotix_ws/src/aloha/scripts/test_sensor_integration.py --check-hdf5 ~/aloha_data/aloha_mobile_pick_object/episode_0.hdf5
"""

import argparse
import sys
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge


# Expected topics and their minimum acceptable Hz
CAMERA_TOPICS = {
    '/cam_high/camera/color/image_rect_raw': {'min_hz': 10, 'qos_depth': 20},
    '/cam_left_wrist/camera/color/image_rect_raw': {'min_hz': 10, 'qos_depth': 20},
    '/cam_right_wrist/camera/color/image_rect_raw': {'min_hz': 10, 'qos_depth': 20},
    '/cam_pov/camera/color/image_raw': {'min_hz': 5, 'qos_depth': 20},
}

ARM_TOPICS = {
    '/follower_left/joint_states': {'min_hz': 50},
    '/follower_right/joint_states': {'min_hz': 50},
    '/leader_left/joint_states': {'min_hz': 50},
    '/leader_right/joint_states': {'min_hz': 50},
}


class SensorTester(Node):
    def __init__(self):
        super().__init__('sensor_tester')
        self.bridge = CvBridge()
        self.results = {}
        self.msg_counts = {}
        self.last_images = {}
        self.subscriptions_list = []

    def test_topics_exist(self):
        """Check that all expected topics are being published."""
        print('\n' + '=' * 60)
        print('TEST 1: Checking expected topics exist')
        print('=' * 60)

        topic_names = [t[0] for t in self.get_topic_names_and_types()]
        all_pass = True

        for topic in list(CAMERA_TOPICS.keys()) + list(ARM_TOPICS.keys()):
            exists = topic in topic_names
            status = 'PASS' if exists else 'FAIL'
            if not exists:
                all_pass = False
            print(f'  [{status}] {topic}')

        self.results['topics_exist'] = all_pass
        return all_pass

    def test_topic_rates(self, duration=5.0):
        """Measure publish rates for all topics."""
        print('\n' + '=' * 60)
        print(f'TEST 2: Measuring topic rates ({duration}s sample)')
        print('=' * 60)

        self.msg_counts = {}
        self.last_images = {}

        # Subscribe to camera topics
        for topic, config in CAMERA_TOPICS.items():
            self.msg_counts[topic] = 0
            self.last_images[topic] = None

            if 'cam_pov' in topic:
                qos = QoSProfile(
                    depth=config['qos_depth'],
                    reliability=ReliabilityPolicy.RELIABLE,
                    durability=DurabilityPolicy.TRANSIENT_LOCAL,
                )
            else:
                qos = config['qos_depth']

            sub = self.create_subscription(
                Image, topic,
                lambda msg, t=topic: self._image_cb(msg, t),
                qos,
            )
            self.subscriptions_list.append(sub)

        # Subscribe to arm topics
        for topic, config in ARM_TOPICS.items():
            self.msg_counts[topic] = 0
            sub = self.create_subscription(
                JointState, topic,
                lambda msg, t=topic: self._count_cb(t),
                10,
            )
            self.subscriptions_list.append(sub)

        # Spin for the measurement duration
        start = time.time()
        while time.time() - start < duration:
            rclpy.spin_once(self, timeout_sec=0.1)

        # Evaluate results
        all_pass = True
        all_topics = {**CAMERA_TOPICS, **ARM_TOPICS}
        for topic, config in all_topics.items():
            count = self.msg_counts.get(topic, 0)
            hz = count / duration
            min_hz = config['min_hz']
            passed = hz >= min_hz
            status = 'PASS' if passed else 'FAIL'
            if not passed:
                all_pass = False
            print(f'  [{status}] {topic}: {hz:.1f} Hz (min: {min_hz} Hz)')

        # Destroy subscriptions
        for sub in self.subscriptions_list:
            self.destroy_subscription(sub)
        self.subscriptions_list.clear()

        self.results['topic_rates'] = all_pass
        return all_pass

    def test_image_content(self):
        """Check that camera images are not all black."""
        print('\n' + '=' * 60)
        print('TEST 3: Checking camera images are not black')
        print('=' * 60)

        all_pass = True
        for topic, img in self.last_images.items():
            if img is None:
                print(f'  [FAIL] {topic}: No image received')
                all_pass = False
                continue

            mean_val = img.mean()
            is_black = mean_val < 1.0
            status = 'FAIL' if is_black else 'PASS'
            if is_black:
                all_pass = False
            print(f'  [{status}] {topic}: mean={mean_val:.1f}, shape={img.shape}')

        self.results['image_content'] = all_pass
        return all_pass

    def test_qos_compatibility(self):
        """Check QoS profiles for known problematic topics."""
        print('\n' + '=' * 60)
        print('TEST 4: Checking QoS compatibility')
        print('=' * 60)

        all_pass = True
        pov_topic = '/cam_pov/camera/color/image_raw'

        # Get publisher info for cam_pov
        pub_info = self.get_publishers_info_by_topic(pov_topic)
        if not pub_info:
            print(f'  [FAIL] {pov_topic}: No publishers found')
            all_pass = False
        else:
            for info in pub_info:
                durability = info.qos_profile.durability
                dur_name = str(durability).split('.')[-1]
                is_transient = 'TRANSIENT_LOCAL' in dur_name.upper()
                status = 'PASS' if is_transient else 'WARN'
                print(f'  [{status}] {pov_topic} publisher durability: {dur_name}')
                if not is_transient:
                    print(f'         Subscriber must use TRANSIENT_LOCAL to match')

        self.results['qos_compatibility'] = all_pass
        return all_pass

    def test_arm_positions(self, duration=2.0):
        """Check that arm joint positions are valid (not all -pi)."""
        print('\n' + '=' * 60)
        print('TEST 5: Checking arm joint positions are valid')
        print('=' * 60)

        self.arm_positions = {}

        for topic in ARM_TOPICS:
            self.arm_positions[topic] = None
            sub = self.create_subscription(
                JointState, topic,
                lambda msg, t=topic: self._joint_cb(msg, t),
                10,
            )
            self.subscriptions_list.append(sub)

        start = time.time()
        while time.time() - start < duration:
            rclpy.spin_once(self, timeout_sec=0.1)

        all_pass = True
        for topic, positions in self.arm_positions.items():
            if positions is None:
                print(f'  [FAIL] {topic}: No joint data received')
                all_pass = False
                continue

            pos = np.array(positions)
            all_negative_pi = np.allclose(pos, -np.pi, atol=0.01)
            status = 'FAIL' if all_negative_pi else 'PASS'
            if all_negative_pi:
                all_pass = False
                print(f'  [{status}] {topic}: All joints at -pi (communication failure!)')
            else:
                print(f'  [{status}] {topic}: positions={np.round(pos, 3)}')

        for sub in self.subscriptions_list:
            self.destroy_subscription(sub)
        self.subscriptions_list.clear()

        self.results['arm_positions'] = all_pass
        return all_pass

    def _image_cb(self, msg, topic):
        self.msg_counts[topic] = self.msg_counts.get(topic, 0) + 1
        try:
            self.last_images[topic] = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            pass

    def _count_cb(self, topic):
        self.msg_counts[topic] = self.msg_counts.get(topic, 0) + 1

    def _joint_cb(self, msg, topic):
        self.arm_positions[topic] = list(msg.position)

    def print_summary(self):
        print('\n' + '=' * 60)
        print('SUMMARY')
        print('=' * 60)
        all_pass = True
        for test_name, passed in self.results.items():
            status = 'PASS' if passed else 'FAIL'
            if not passed:
                all_pass = False
            print(f'  [{status}] {test_name}')

        print()
        if all_pass:
            print('  ALL TESTS PASSED')
        else:
            print('  SOME TESTS FAILED - see details above')
        print('=' * 60)
        return all_pass


def test_hdf5(filepath):
    """Validate a recorded HDF5 episode file."""
    import h5py

    print('\n' + '=' * 60)
    print(f'HDF5 VALIDATION: {filepath}')
    print('=' * 60)

    try:
        f = h5py.File(filepath, 'r')
    except Exception as e:
        print(f'  [FAIL] Cannot open file: {e}')
        return False

    all_pass = True

    # Check datasets exist
    expected = ['observations/qpos', 'action', 'compress_len']
    for ds in expected:
        exists = ds in f or ds.split('/')[-1] in f.get(ds.rsplit('/', 1)[0], {})
        if ds in f:
            print(f'  [PASS] Dataset {ds} exists, shape={f[ds].shape}')
        else:
            print(f'  [FAIL] Dataset {ds} missing')
            all_pass = False

    # Check camera images
    camera_names = ['cam_high', 'cam_left_wrist', 'cam_right_wrist', 'cam_pov']
    compress_len = f['compress_len'][:]

    for i, cam in enumerate(camera_names):
        cam_path = f'observations/images/{cam}'
        if cam_path.split('/')[-1] not in f.get('observations/images', {}):
            print(f'  [FAIL] {cam}: not found in HDF5')
            all_pass = False
            continue

        # Check a few frames
        num_timesteps = f[cam_path].shape[0]
        test_frames = [0, num_timesteps // 4, num_timesteps // 2, num_timesteps - 1]
        black_count = 0

        for t in test_frames:
            raw = f[cam_path][t]
            length = int(compress_len[i][t])
            img = cv2.imdecode(np.frombuffer(raw[:length], dtype='uint8'), cv2.IMREAD_COLOR)
            if img is None or img.mean() < 1.0:
                black_count += 1

        if black_count == len(test_frames):
            print(f'  [FAIL] {cam}: All sampled frames are black')
            all_pass = False
        elif black_count > 0:
            print(f'  [WARN] {cam}: {black_count}/{len(test_frames)} sampled frames are black')
        else:
            print(f'  [PASS] {cam}: {num_timesteps} frames, all sampled frames have content')

    # Check arm data validity
    qpos = f['observations/qpos'][:]
    left_std = qpos[:, :7].std(axis=0)
    right_std = qpos[:, 7:].std(axis=0)

    left_stuck = np.all(left_std < 1e-6)
    right_stuck = np.all(right_std < 1e-6)

    if left_stuck:
        print(f'  [FAIL] Left arm qpos has zero variance (stuck/disconnected)')
        all_pass = False
    else:
        print(f'  [PASS] Left arm qpos has variance (arm moved)')

    if right_stuck:
        print(f'  [FAIL] Right arm qpos has zero variance (stuck/disconnected)')
        all_pass = False
    else:
        print(f'  [PASS] Right arm qpos has variance (arm moved)')

    f.close()

    print()
    if all_pass:
        print('  HDF5 VALIDATION PASSED')
    else:
        print('  HDF5 VALIDATION FAILED - see details above')
    print('=' * 60)
    return all_pass


def main():
    parser = argparse.ArgumentParser(description='ALOHA sensor integration test')
    parser.add_argument(
        '--check-hdf5', type=str, default=None,
        help='Path to an HDF5 episode file to validate'
    )
    parser.add_argument(
        '--skip-ros', action='store_true',
        help='Skip ROS topic tests (only run HDF5 validation)'
    )
    args = parser.parse_args()

    all_pass = True

    if not args.skip_ros:
        rclpy.init()
        tester = SensorTester()

        try:
            tester.test_topics_exist()
            tester.test_topic_rates(duration=5.0)
            tester.test_image_content()
            tester.test_qos_compatibility()
            tester.test_arm_positions(duration=2.0)
            all_pass = tester.print_summary()
        finally:
            tester.destroy_node()
            rclpy.shutdown()

    if args.check_hdf5:
        hdf5_pass = test_hdf5(args.check_hdf5)
        all_pass = all_pass and hdf5_pass

    sys.exit(0 if all_pass else 1)


if __name__ == '__main__':
    main()
