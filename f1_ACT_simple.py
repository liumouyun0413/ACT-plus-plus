#!/usr/bin/env python3
"""F1 ACT inference using the ROS 2 interfaces documented by the F1 system.

Required F1 topics:
  observations:
    /camera/{left_wrist,right_wrist,head}/color/image_raw/compressed
    /hal/joint_states
    /motion_ctl/gripper/{left,right}/state
  commands:
    /motion_ctl/joint_ctl
    /motion_ctl/gripper/{left,right}

The ACT data contract is:
  qpos/action[0:14] = arm_l_j1..j7, arm_r_j1..j7 in radians
  qpos/action[14:16] = left/right gripper positions in [0, 100]
"""

from __future__ import annotations

import argparse
import os
import pickle
import signal
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

if "--gpu" in sys.argv:
    i = sys.argv.index("--gpu")
    if i + 1 < len(sys.argv):
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[i + 1]

import torch
import rclpy
from control_msgs.msg import GripperCommand
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, JointState

from constants import REAL_TASK_CONFIGS
from policy import ACTPolicy

CAMERA_TOPICS = {
    "hand_left_color": "/camera/left_wrist/color/image_raw/compressed",
    "hand_right_color": "/camera/right_wrist/color/image_raw/compressed",
    "head_color": "/camera/head/color/image_raw/compressed",
}
ARM_NAMES = [f"arm_l_j{i}" for i in range(1, 8)] + [f"arm_r_j{i}" for i in range(1, 8)]
IMG_SIZE = (480, 640)
# F1 manual limits in degrees, contracted by 1 degree for safety.
ARM_LOW_DEG = np.array([-177, -119, -177, -144, -177, -59, -89] * 2, np.float32)
ARM_HIGH_DEG = np.array([177, 119, 177, 59, 177, 59, 89] * 2, np.float32)
ARM_LOW_RAD = np.deg2rad(ARM_LOW_DEG)
ARM_HIGH_RAD = np.deg2rad(ARM_HIGH_DEG)
GRIPPER_LOW, GRIPPER_HIGH = 0.0, 100.0


class F1ROSInterface(Node):
    def __init__(self, camera_names, sensor_timeout):
        super().__init__("f1_act_inference")
        self.camera_names = camera_names
        self.sensor_timeout = sensor_timeout
        self._lock = threading.Lock()
        self._images = {}
        self._image_stamps = {}
        self._joint = None
        self._joint_stamp = 0.0
        self._grippers = {"left": None, "right": None}
        self._gripper_stamps = {"left": 0.0, "right": 0.0}
        reliable_volatile = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=50,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        reliable_transient = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        for name in camera_names:
            # F1 wrist camera publishers are transient_local; the Orbbec head
            # camera is volatile. Match the publisher QoS verified on hardware.
            image_qos = reliable_volatile if name == "head_color" else reliable_transient
            self.create_subscription(
                CompressedImage, CAMERA_TOPICS[name],
                lambda msg, camera=name: self._on_image(camera, msg), image_qos,
            )
        self.create_subscription(JointState, "/hal/joint_states", self._on_joint, reliable_volatile)
        self.create_subscription(JointState, "/motion_ctl/gripper/left/state", lambda m: self._on_gripper("left", m), reliable_volatile)
        self.create_subscription(JointState, "/motion_ctl/gripper/right/state", lambda m: self._on_gripper("right", m), reliable_volatile)
        self.arm_pub = self.create_publisher(JointState, "/motion_ctl/joint_ctl", 20)
        self.left_gripper_pub = self.create_publisher(GripperCommand, "/motion_ctl/gripper/left", 20)
        self.right_gripper_pub = self.create_publisher(GripperCommand, "/motion_ctl/gripper/right", 20)

    @staticmethod
    def _stamp(msg):
        stamp = msg.header.stamp
        value = float(stamp.sec) + float(stamp.nanosec) / 1e9
        return value if value > 0 else time.time()

    def _on_image(self, name, msg):
        image = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            self.get_logger().error(f"JPEG decode failed: {name}")
            return
        # Match training: model receives RGB from every camera.
        rgb = cv2.cvtColor(cv2.resize(image, (IMG_SIZE[1], IMG_SIZE[0])), cv2.COLOR_BGR2RGB)
        with self._lock:
            self._images[name] = rgb
            self._image_stamps[name] = self._stamp(msg)

    def _on_joint(self, msg):
        with self._lock:
            self._joint = msg
            self._joint_stamp = self._stamp(msg)

    def _on_gripper(self, side, msg):
        if not msg.position:
            return
        with self._lock:
            self._grippers[side] = float(msg.position[0])
            self._gripper_stamps[side] = self._stamp(msg)

    def _check_age(self, label, stamp):
        age = time.time() - stamp
        if age > self.sensor_timeout:
            raise RuntimeError(f"stale {label}: {age:.3f}s > {self.sensor_timeout:.3f}s")

    def observation(self):
        with self._lock:
            if self._joint is None or any(x is None for x in self._grippers.values()):
                raise RuntimeError("joint/gripper state not ready")
            if any(name not in self._images for name in self.camera_names):
                raise RuntimeError("camera image not ready")
            joint = self._joint
            images = {name: self._images[name].copy() for name in self.camera_names}
            image_stamps = dict(self._image_stamps)
            grippers = dict(self._grippers)
            gripper_stamps = dict(self._gripper_stamps)
            joint_stamp = self._joint_stamp
        self._check_age("joint state", joint_stamp)
        for name, stamp in image_stamps.items():
            self._check_age(name, stamp)
        for side, stamp in gripper_stamps.items():
            self._check_age(f"{side} gripper", stamp)
        if max(image_stamps.values()) - min(image_stamps.values()) > self.sensor_timeout:
            raise RuntimeError("camera timestamps are not synchronized")
        positions = dict(zip(joint.name, joint.position))
        missing = [name for name in ARM_NAMES if name not in positions]
        if missing:
            raise RuntimeError(f"missing F1 joints: {missing}")
        # F1 ROS joint state/control uses degrees; ACT uses radians.
        qpos = np.zeros(16, np.float32)
        qpos[:14] = np.deg2rad([positions[name] for name in ARM_NAMES])
        qpos[14] = grippers["left"]
        qpos[15] = grippers["right"]
        return images, qpos

    def publish_action(self, action, max_effort):
        arm = JointState()
        arm.header.stamp = self.get_clock().now().to_msg()
        arm.header.frame_id = "base_link"
        arm.name = ARM_NAMES
        arm.position = np.rad2deg(action[:14]).astype(float).tolist()
        self.arm_pub.publish(arm)
        left = GripperCommand(position=float(action[14]), max_effort=max_effort)
        right = GripperCommand(position=float(action[15]), max_effort=max_effort)
        self.left_gripper_pub.publish(left)
        self.right_gripper_pub.publish(right)

    def publish_arm_only(self, action):
        """Hold the arm without repeatedly changing the gripper command."""
        arm = JointState()
        arm.header.stamp = self.get_clock().now().to_msg()
        arm.header.frame_id = "base_link"
        arm.name = ARM_NAMES
        arm.position = np.rad2deg(action[:14]).astype(float).tolist()
        self.arm_pub.publish(arm)


class ROSSpinThread:
    def __init__(self, node):
        self.node = node
        self.thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        rclpy.shutdown()
        self.thread.join(timeout=2)


def load_policy(ckpt_dir, ckpt_name, device):
    with open(Path(ckpt_dir) / "config.pkl", "rb") as f:
        config = pickle.load(f)
    policy = ACTPolicy(config["policy_config"])
    policy.deserialize(torch.load(Path(ckpt_dir) / ckpt_name, map_location=device))
    return policy.to(device).eval(), config["policy_config"]


def load_stats(ckpt_dir, device):
    with open(Path(ckpt_dir) / "dataset_stats.pkl", "rb") as f:
        stats = pickle.load(f)
    required = ("qpos_mean", "qpos_std", "action_mean", "action_std")
    missing = [key for key in required if key not in stats]
    if missing:
        raise KeyError(f"dataset stats missing required keys: {missing}")
    return {
        key: torch.as_tensor(stats[key], dtype=torch.float32, device=device)
        for key in required
    }


def infer(policy, interface, camera_names, stats, device):
    started = time.perf_counter()
    images, qpos = interface.observation()
    image_np = np.stack([images[name].transpose(2, 0, 1) for name in camera_names])
    image = torch.from_numpy(image_np).float().to(device).unsqueeze(0) / 255.0
    qpos_t = torch.from_numpy(qpos).float().to(device).unsqueeze(0)
    qpos_t = (qpos_t - stats["qpos_mean"]) / stats["qpos_std"]
    with torch.inference_mode():
        result = policy(qpos_t, image)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = result * stats["action_std"] + stats["action_mean"]
    chunk = result[0].cpu().numpy()
    if not np.isfinite(chunk).all():
        raise RuntimeError("policy produced non-finite actions")
    return chunk, qpos, (time.perf_counter() - started) * 1000


def clamp_action(action, previous, max_joint_delta):
    value = np.asarray(action, np.float32).copy()
    value[:14] = np.clip(value[:14], ARM_LOW_RAD, ARM_HIGH_RAD)
    value[14:] = np.clip(value[14:], GRIPPER_LOW, GRIPPER_HIGH)
    if previous is not None:
        value[:14] = previous[:14] + np.clip(
            value[:14] - previous[:14], -max_joint_delta, max_joint_delta
        )
    return value


def execute_chunk(interface, chunk, previous, action_period, command_rate, max_joint_delta, max_effort, should_stop=None):
    """Skip chunk[0], then execute prev_action -> chunk[1] -> ... at command_rate."""
    interpolation_steps = max(1, round(action_period * command_rate))
    period = 1.0 / command_rate
    command_count = 0
    for index in range(1, len(chunk)):
        start = previous.copy() if index == 1 else chunk[index - 1]
        target = chunk[index]
        for step in range(1, interpolation_steps + 1):
            if should_stop is not None and should_stop():
                return previous, command_count
            tick = time.monotonic()
            raw = start + (target - start) * (step / interpolation_steps)
            command = clamp_action(raw, previous, max_joint_delta)
            interface.publish_action(command, max_effort)
            previous = command
            command_count += 1
            remaining = period - (time.monotonic() - tick)
            if remaining > 0:
                time.sleep(remaining)
    return previous, command_count


def infer_while_holding(
    policy, interface, camera_names, stats, device, previous,
    command_rate, should_stop,
):
    """Synchronously obtain one chunk while the main loop holds the arm at 100Hz.

    The worker exists only so the main thread can keep publishing hold commands.
    It is joined before execution starts and never overlaps chunk execution.
    """
    result = {"value": None, "error": None}
    finished = threading.Event()

    def worker():
        try:
            result["value"] = infer(
                policy, interface, camera_names, stats, device
            )
        except Exception as exc:
            result["error"] = exc
        finally:
            finished.set()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    period = 1.0 / command_rate
    hold_count = 0
    while not finished.is_set() and not should_stop():
        tick = time.monotonic()
        interface.publish_arm_only(previous)
        hold_count += 1
        remaining = period - (time.monotonic() - tick)
        if remaining > 0:
            finished.wait(remaining)
    thread.join()
    if result["error"] is not None:
        raise RuntimeError("synchronous chunk inference failed") from result["error"]
    return result["value"], hold_count


def main():
    p = argparse.ArgumentParser(description="F1 ACT inference over ROS 2")
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--ckpt_name", default="policy_best.ckpt")
    p.add_argument("--task_name", default="fold_clothes_f1")
    p.add_argument("--max_chunks", type=int, default=200)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--action_period", type=float, default=1 / 30, help="ACT sample period; fold_clothes is 30Hz")
    p.add_argument("--command_rate", type=float, default=100.0)
    p.add_argument("--max_joint_delta", type=float, default=np.deg2rad(0.5), help="max arm delta per command, radians")
    p.add_argument("--sensor_timeout", type=float, default=0.2)
    p.add_argument("--max_effort", type=float, default=10.0)
    p.add_argument("--warmup_runs", type=int, default=3)
    p.add_argument("--dry_run", action="store_true", help="run inference without publishing controls")
    args = p.parse_args()

    if args.max_chunks < 1 or args.warmup_runs < 0:
        raise ValueError("max_chunks must be >= 1 and warmup_runs must be >= 0")

    if args.task_name not in REAL_TASK_CONFIGS:
        raise KeyError(f"unknown task: {args.task_name}")
    camera_names = REAL_TASK_CONFIGS[args.task_name]["camera_names"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, policy_config = load_policy(args.ckpt_dir, args.ckpt_name, device)
    stats = load_stats(args.ckpt_dir, device)
    if policy_config["camera_names"] != camera_names:
        raise ValueError("checkpoint camera order does not match F1 task config")

    rclpy.init()
    interface = F1ROSInterface(camera_names, args.sensor_timeout)
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    with ROSSpinThread(interface):
        deadline = time.time() + 10
        while True:
            try:
                _, initial_qpos = interface.observation()
                break
            except RuntimeError as exc:
                if time.time() >= deadline:
                    raise RuntimeError(f"F1 sensor startup failed: {exc}")
                time.sleep(0.1)
        print("F1 sensors ready. Joint order and units validated.")
        warmup_times = []
        for _ in range(args.warmup_runs):
            _, _, inference_ms = infer(
                policy, interface, camera_names, stats, device
            )
            warmup_times.append(inference_ms)
        if warmup_times:
            print(
                "GPU warmup completed: "
                + ", ".join(f"{value:.1f}ms" for value in warmup_times)
            )
        if args.dry_run:
            chunk, _, inference_ms = infer(
                policy, interface, camera_names, stats, device
            )
            print(
                f"dry-run OK: chunk={chunk.shape}, "
                f"range=[{chunk.min():.3f}, {chunk.max():.3f}], "
                f"inference={inference_ms:.1f}ms"
            )
            return
        input("Press Enter to start F1 arm control, or Ctrl-C to abort: ")
        previous = initial_qpos
        for index in range(args.max_chunks):
            if stopping:
                break
            (chunk, observed_qpos, inference_ms), hold_count = infer_while_holding(
                policy, interface, camera_names, stats, device, previous,
                args.command_rate, lambda: stopping,
            )
            if stopping:
                break
            boundary_error_deg = np.max(
                np.abs(np.rad2deg(chunk[0, :14] - previous[:14]))
            )
            observation_error_deg = np.max(
                np.abs(np.rad2deg(chunk[0, :14] - observed_qpos[:14]))
            )
            previous, command_count = execute_chunk(
                interface, chunk, previous, args.action_period, args.command_rate,
                args.max_joint_delta, args.max_effort, lambda: stopping,
            )
            print(
                f"chunk {index + 1}/{args.max_chunks} completed; "
                f"hold={hold_count}, inference={inference_ms:.1f}ms, "
                f"skip=1, commands={command_count}, "
                f"chunk0-prev={boundary_error_deg:.2f}deg, "
                f"chunk0-qpos={observation_error_deg:.2f}deg"
            )
        # Hold only the final arm pose briefly; grippers keep their final command.
        for _ in range(50):
            interface.publish_arm_only(previous)
            time.sleep(1 / args.command_rate)
    interface.destroy_node()


if __name__ == "__main__":
    main()
