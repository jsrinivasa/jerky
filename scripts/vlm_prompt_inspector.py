#!/usr/bin/env python3

import argparse
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor

from aloha.constants import DATA_DIR
from aloha.constants import FOLLOWER_GRIPPER_JOINT_CLOSE, FOLLOWER_GRIPPER_JOINT_OPEN, START_ARM_POSE
from aloha.vlm_controller import VLMAction
from aloha.robot_utils import get_arm_joint_positions, move_arms, move_grippers


_CAM_SOURCE_BY_LOGICAL_NAME = {
    'cam_left_wrist': 'cam_high',
    'cam_high': 'cam_right_wrist',
    'cam_right_wrist': 'cam_left_wrist',
}


def _remap_cam_name(cam_name: str) -> str:
    return _CAM_SOURCE_BY_LOGICAL_NAME.get(cam_name, cam_name)


def _remap_images_for_cams(images: Dict[str, Optional[np.ndarray]], cams: Sequence[str]) -> Dict[str, Optional[np.ndarray]]:
    return {cam: images.get(_remap_cam_name(cam)) for cam in cams}


class _RosSpinThread:
    def __init__(self, executor: SingleThreadedExecutor):
        self._executor = executor
        self._stop_evt = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while rclpy.ok() and not self._stop_evt.is_set():
            try:
                self._executor.spin_once(timeout_sec=0.1)
            except Exception:
                break


def _now_str() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _resize_keep_aspect(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == target_h:
        return img
    scale = float(target_h) / float(h)
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)


def _render_missing(target_h: int, target_w: int, label: str) -> np.ndarray:
    img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    cv2.putText(img, label, (10, target_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return img


def _overlay_label(img: np.ndarray, label: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 32), (0, 0, 0), thickness=-1)
    cv2.putText(out, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out


def _tile_images(images: Dict[str, Optional[np.ndarray]], cam_names: Sequence[str], target_h: int = 360) -> np.ndarray:
    tiles: List[np.ndarray] = []
    widths: List[int] = []

    for cam in cam_names:
        img = images.get(cam)
        if img is None:
            tile = _render_missing(target_h, int(target_h * 4 / 3), f"{cam}: MISSING")
        else:
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            tile = _resize_keep_aspect(img, target_h)
        tile = _overlay_label(tile, cam)
        tiles.append(tile)
        widths.append(tile.shape[1])

    max_w = max(widths) if widths else int(target_h * 4 / 3)
    padded: List[np.ndarray] = []
    for tile in tiles:
        if tile.shape[1] < max_w:
            pad = np.zeros((tile.shape[0], max_w - tile.shape[1], 3), dtype=tile.dtype)
            tile = np.concatenate([tile, pad], axis=1)
        padded.append(tile)

    return np.concatenate(padded, axis=1) if padded else np.zeros((target_h, target_h, 3), dtype=np.uint8)


def _wait_for_images(controller: Any, required_cams: Sequence[str], timeout_s: float = 5.0) -> bool:
    start = time.time()
    while time.time() - start < timeout_s and rclpy.ok():
        images = controller.image_recorder.get_images()
        if all(images.get(c) is not None for c in required_cams):
            return True
        time.sleep(0.05)
    return False


def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _create_controller(args: argparse.Namespace):
    from aloha.vlm_llava import LLaVAController

    return LLaVAController(
        model_path=args.model_path,
        device=args.device,
        load_in_8bit=args.load_8bit,
        load_in_4bit=args.load_4bit,
        verbose_logging=False,
        enable_base=False,
        enable_arms=args.enable_arms,
        control_frequency=args.frequency,
        node=None,
    )


def _call_llava(controller: Any, prompt: str, image_bgr: np.ndarray) -> str:
    from PIL import Image
    import torch

    if controller.model is None or controller.processor is None:
        return "Model not loaded"

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image_rgb)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    prompt_text = controller.processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = controller.processor(images=image, text=prompt_text, return_tensors="pt").to(controller.device)

    with torch.no_grad():
        output = controller.model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
        )

    generated_text = controller.processor.decode(output[0], skip_special_tokens=True)
    if "ASSISTANT:" in generated_text:
        return generated_text.split("ASSISTANT:")[-1].strip()
    return generated_text.strip()


def _clamp(x: float, lo: float, hi: float) -> float:
    return min(max(x, lo), hi)


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    try:
        if '```json' in text:
            json_str = text.split('```json')[1].split('```')[0]
            return json.loads(json_str)
        start = text.find('{')
        end = text.rfind('}')
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
    except Exception:
        return None
    return None


def _get_follower_bot(controller: Any, side: str):
    if side == 'left':
        return controller.follower_bot_left
    if side == 'right':
        return controller.follower_bot_right
    raise ValueError(f"Invalid side: {side}")


def _tool_spec_text() -> str:
    return (
        "You can control the robot arms by calling tools. Respond ONLY with valid JSON.\n"
        "Schema:\n"
        "{\n"
        "  \"tool_calls\": [\n"
        "    {\"name\": \"arm_nudge_joint\", \"arguments\": {\"side\": \"left|right\", \"dq\": [6 floats], \"moving_time\": float}},\n"
        "    {\"name\": \"arm_go_named_pose\", \"arguments\": {\"side\": \"left|right\", \"name\": \"start|sleep\", \"moving_time\": float}},\n"
        "    {\"name\": \"gripper_open\", \"arguments\": {\"side\": \"left|right\", \"moving_time\": float}},\n"
        "    {\"name\": \"gripper_close\", \"arguments\": {\"side\": \"left|right\", \"moving_time\": float}},\n"
        "    {\"name\": \"gripper_set\", \"arguments\": {\"side\": \"left|right\", \"q\": float, \"moving_time\": float}}\n"
        "  ],\n"
        "  \"done\": false,\n"
        "  \"status\": \"short description of what you did\"\n"
        "}\n"
        "Rules:\n"
        "- Use SMALL joint nudges. dq is clamped per joint to +/-0.12 rad.\n"
        "- moving_time is clamped to [0.1, 2.0] seconds.\n"
        "- gripper q is clamped to follower limits.\n"
        "- If you are unsure, do a small nudge and re-check.\n"
    )


def _build_closed_loop_prompt(task: str, observation: Dict[str, Any], history: List[Dict[str, Any]]) -> str:
    state = observation.get('robot_state', {})
    left_q = state.get('left_arm_qpos')
    right_q = state.get('right_arm_qpos')
    left_g = state.get('left_gripper_qpos')
    right_g = state.get('right_gripper_qpos')
    step = len(history) + 1

    recent = ""
    if history:
        recent = "\nRecent tool calls (most recent last):\n"
        for h in history[-5:]:
            tc = h.get('tool_calls')
            if tc is not None:
                recent += f"- {json.dumps(tc)}\n"

    return (
        f"Task: {task}\n"
        f"Step: {step}\n"
        f"Follower arm state (radians):\n"
        f"- left_arm_qpos: {left_q}\n"
        f"- right_arm_qpos: {right_q}\n"
        f"Follower gripper state:\n"
        f"- left_gripper_qpos: {left_g}\n"
        f"- right_gripper_qpos: {right_g}\n"
        f"\n{recent}\n"
        f"\n{_tool_spec_text()}"
    )


def _execute_tool_calls(controller: Any, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    executed: List[Dict[str, Any]] = []
    for call in tool_calls:
        name = str(call.get('name', '')).strip()
        args = call.get('arguments', {}) or {}

        side = str(args.get('side', '')).strip().lower()
        if side not in ['left', 'right']:
            executed.append({'name': name, 'ok': False, 'error': f"invalid side: {side}"})
            continue

        bot = _get_follower_bot(controller, side)

        if name == 'arm_nudge_joint':
            dq = args.get('dq')
            if not (isinstance(dq, list) and len(dq) == 6):
                executed.append({'name': name, 'ok': False, 'error': 'dq must be length-6 list'})
                continue
            dq = [float(_clamp(float(x), -0.12, 0.12)) for x in dq]
            moving_time = float(_clamp(float(args.get('moving_time', 0.4)), 0.1, 2.0))
            q_curr = list(get_arm_joint_positions(bot))
            q_tgt = [qc + d for qc, d in zip(q_curr, dq)]
            move_arms([bot], [q_tgt], moving_time=moving_time)
            executed.append({'name': name, 'ok': True, 'side': side, 'dq': dq, 'moving_time': moving_time})
            continue

        if name == 'arm_go_named_pose':
            pose_name = str(args.get('name', '')).strip().lower()
            moving_time = float(_clamp(float(args.get('moving_time', 2.0)), 0.1, 2.0))
            if pose_name == 'start':
                q_tgt = list(START_ARM_POSE[:6] if side == 'left' else START_ARM_POSE[8:14])
            elif pose_name == 'sleep':
                q_tgt = list(bot.arm.group_info.joint_sleep_positions)
            else:
                executed.append({'name': name, 'ok': False, 'error': f"invalid pose name: {pose_name}"})
                continue
            move_arms([bot], [q_tgt], moving_time=moving_time)
            executed.append({'name': name, 'ok': True, 'side': side, 'name': pose_name, 'moving_time': moving_time})
            continue

        if name in ['gripper_open', 'gripper_close', 'gripper_set']:
            moving_time = float(_clamp(float(args.get('moving_time', 0.2)), 0.1, 2.0))
            if name == 'gripper_open':
                q = float(FOLLOWER_GRIPPER_JOINT_OPEN)
            elif name == 'gripper_close':
                q = float(FOLLOWER_GRIPPER_JOINT_CLOSE)
            else:
                q = float(args.get('q', FOLLOWER_GRIPPER_JOINT_OPEN))
                q = float(_clamp(q, float(FOLLOWER_GRIPPER_JOINT_CLOSE), float(FOLLOWER_GRIPPER_JOINT_OPEN)))
            move_grippers([bot], [q], moving_time=moving_time)
            executed.append({'name': name, 'ok': True, 'side': side, 'q': q, 'moving_time': moving_time})
            continue

        executed.append({'name': name, 'ok': False, 'error': 'unknown tool'})

    return executed


def main() -> int:
    parser = argparse.ArgumentParser(
        description='LLaVA prompt inspector (interactive): capture camera images, send prompt to local LLaVA, log outputs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--model-path', type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--load-8bit', action='store_true')
    parser.add_argument('--load-4bit', action='store_true', default=True)

    parser.add_argument('--frequency', type=float, default=5.0)
    parser.add_argument('--log-root', type=str, default=str(Path(DATA_DIR) / 'vlm_prompt_inspector'))
    parser.add_argument('--no-display', action='store_true', help='Do not open OpenCV windows')
    parser.add_argument('--save-images', action='store_true', default=True)
    parser.add_argument('--vlm-image-cam', type=str, default='cam_high', help='Which camera image to send to LLaVA')
    parser.add_argument('--show-cams', nargs='+', default=['cam_left_wrist', 'cam_high', 'cam_right_wrist'], help='Which camera(s) to display/log')
    parser.add_argument('--image-wait-timeout', type=float, default=5.0)

    parser.add_argument('--enable-arms', action='store_true', help='Enable follower arm control (required for closed-loop)')
    parser.add_argument('--closed-loop', action='store_true', help='Run closed-loop tool-calling control loop after entering a task')
    parser.add_argument('--max-steps', type=int, default=25)

    args = parser.parse_args()

    if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
        args.no_display = True

    rclpy.init()

    controller = None
    executor = None
    spinner = None

    session_dir = Path(args.log_root) / _now_str()
    _ensure_dir(session_dir)
    images_dir = session_dir / 'images'
    _ensure_dir(images_dir)
    log_path = session_dir / 'log.jsonl'

    try:
        controller = _create_controller(args)

        executor = SingleThreadedExecutor()
        executor.add_node(controller.node)
        spinner = _RosSpinThread(executor)
        spinner.start()

        required_source_cams = sorted({
            _remap_cam_name(c) for c in list(args.show_cams) + [args.vlm_image_cam]
        })

        if not _wait_for_images(controller, required_source_cams, timeout_s=args.image_wait_timeout):
            print(f"Warning: did not receive all camera frames within {args.image_wait_timeout:.1f}s")

        print("\n" + "=" * 80)
        print("VLM Prompt Inspector")
        print("=" * 80)
        print("Model: llava (local)")
        print(f"Model path: {args.model_path}")
        print(f"Device: {args.device}")
        print(f"VLM image cam: {args.vlm_image_cam}")
        print(f"Session dir: {session_dir}")
        print("Type a prompt and press Enter.")
        print("Commands: 'quit' / 'exit' / 'q'")

        turn_idx = 0
        history: List[Dict[str, Any]] = []

        while rclpy.ok():
            try:
                user_text = input("prompt> ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_text:
                continue
            if user_text.lower() in ['quit', 'exit', 'q']:
                break

            if args.closed_loop:
                if not args.enable_arms:
                    print("Error: --closed-loop requires --enable-arms")
                    continue

                task = user_text
                step_history: List[Dict[str, Any]] = []

                for step_idx in range(args.max_steps):
                    turn_idx += 1

                    observation = controller.get_observation()
                    images = observation.get('images', {})

                    view_images = _remap_images_for_cams(images, args.show_cams)

                    tiled = None
                    if not args.no_display or args.save_images:
                        tiled = _tile_images(view_images, args.show_cams, target_h=360)
                        if not args.no_display:
                            cv2.imshow('VLM Prompt Inspector', tiled)
                            cv2.waitKey(1)

                    saved_images: Dict[str, str] = {}
                    if args.save_images:
                        for cam in args.show_cams:
                            img = view_images.get(cam)
                            if img is None:
                                continue
                            out_path = images_dir / f"{turn_idx:04d}_{cam}.jpg"
                            cv2.imwrite(str(out_path), img)
                            saved_images[cam] = str(out_path)
                        if tiled is not None:
                            tiled_path = images_dir / f"{turn_idx:04d}_tiled.jpg"
                            cv2.imwrite(str(tiled_path), tiled)
                            saved_images['tiled'] = str(tiled_path)

                    image_for_vlm = images.get(_remap_cam_name(args.vlm_image_cam))
                    if image_for_vlm is None:
                        llava_prompt = _build_closed_loop_prompt(task, observation, step_history)
                        raw_output = f"No image available for {args.vlm_image_cam}"
                    else:
                        llava_prompt = _build_closed_loop_prompt(task, observation, step_history)
                        raw_output = _call_llava(controller, llava_prompt, image_for_vlm)

                    parsed = _extract_json_object(raw_output)
                    tool_calls = []
                    done = False
                    status = None
                    if isinstance(parsed, dict):
                        tool_calls = parsed.get('tool_calls') or []
                        done = bool(parsed.get('done', False))
                        status = parsed.get('status')

                    executed = []
                    if isinstance(tool_calls, list) and tool_calls:
                        executed = _execute_tool_calls(controller, tool_calls)
                    else:
                        executed = [{'name': None, 'ok': False, 'error': 'no tool_calls parsed'}]

                    step_history.append({'tool_calls': tool_calls, 'executed': executed, 'raw_output': raw_output})

                    print("\n" + "-" * 80)
                    print(f"Turn: {turn_idx}  (closed-loop step {step_idx + 1}/{args.max_steps})")
                    print(f"Task: {task}")
                    if status:
                        print(f"Status: {status}")
                    print(f"\n[VLM IMAGE SENT] {args.vlm_image_cam}")
                    print("\n[VLM RAW OUTPUT]\n" + raw_output)
                    print("\n[EXECUTED]\n" + json.dumps(executed, indent=2))
                    print("-" * 80 + "\n")

                    entry = {
                        'turn': turn_idx,
                        'time_unix': time.time(),
                        'mode': 'closed_loop',
                        'task': task,
                        'step_idx': step_idx,
                        'model': 'llava',
                        'model_path': args.model_path,
                        'device': args.device,
                        'vlm_prompt': llava_prompt,
                        'vlm_image_sent': args.vlm_image_cam,
                        'vlm_raw_output': raw_output,
                        'parsed': parsed,
                        'executed': executed,
                        'saved_images': saved_images,
                    }
                    with log_path.open('a', encoding='utf-8') as f:
                        f.write(json.dumps(entry) + "\n")

                    history.append({'step': turn_idx - 1, 'action': VLMAction(reasoning=str(raw_output))})

                    if done:
                        break

                continue

            turn_idx += 1

            observation = controller.get_observation()
            images = observation.get('images', {})

            view_images = _remap_images_for_cams(images, args.show_cams)

            tiled = None
            if not args.no_display or args.save_images:
                tiled = _tile_images(view_images, args.show_cams, target_h=360)
                if not args.no_display:
                    cv2.imshow('VLM Prompt Inspector', tiled)
                    cv2.waitKey(1)

            saved_images: Dict[str, str] = {}
            if args.save_images:
                for cam in args.show_cams:
                    img = view_images.get(cam)
                    if img is None:
                        continue
                    out_path = images_dir / f"{turn_idx:04d}_{cam}.jpg"
                    cv2.imwrite(str(out_path), img)
                    saved_images[cam] = str(out_path)

                if tiled is not None:
                    tiled_path = images_dir / f"{turn_idx:04d}_tiled.jpg"
                    cv2.imwrite(str(tiled_path), tiled)
                    saved_images['tiled'] = str(tiled_path)

            vlm_prompt: str
            raw_output: str
            parsed_action: Optional[Dict[str, Any]] = None

            image_for_vlm = images.get(_remap_cam_name(args.vlm_image_cam))
            if image_for_vlm is None:
                vlm_prompt = controller._create_robot_control_prompt(user_text, observation, history)
                raw_output = f"No image available for {args.vlm_image_cam}"
            else:
                vlm_prompt = controller._create_robot_control_prompt(user_text, observation, history)
                raw_output = _call_llava(controller, vlm_prompt, image_for_vlm)

            action = controller._parse_llava_output(raw_output, observation, user_text)
            parsed_action = {
                'base_action': getattr(action, 'base_action', None).tolist() if hasattr(getattr(action, 'base_action', None), 'tolist') else None,
                'confidence': _safe_float(getattr(action, 'confidence', None)),
                'done': bool(getattr(action, 'done', False)),
                'reasoning': str(getattr(action, 'reasoning', '')),
            }

            print("\n" + "-" * 80)
            print(f"Turn: {turn_idx}")
            print(f"User prompt: {user_text}")
            print("\n[VLM PROMPT]\n" + vlm_prompt)
            print(f"\n[VLM IMAGE SENT] {args.vlm_image_cam}")
            print("\n[VLM RAW OUTPUT]\n" + raw_output)
            if parsed_action is not None:
                print("\n[PARSED ACTION]\n" + json.dumps(parsed_action, indent=2))
            print("-" * 80 + "\n")

            entry = {
                'turn': turn_idx,
                'time_unix': time.time(),
                'model': 'llava',
                'model_path': args.model_path,
                'device': args.device,
                'user_text': user_text,
                'vlm_prompt': vlm_prompt,
                'vlm_image_sent': args.vlm_image_cam,
                'vlm_raw_output': raw_output,
                'parsed_action': parsed_action,
                'saved_images': saved_images,
            }

            with log_path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(entry) + "\n")

            history.append(
                {
                    'step': turn_idx - 1,
                    'action': VLMAction(reasoning=str(raw_output)),
                }
            )

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        try:
            if spinner is not None:
                spinner.stop()
        except Exception:
            pass

        try:
            if controller is not None:
                controller.shutdown()
        except Exception:
            pass

        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
