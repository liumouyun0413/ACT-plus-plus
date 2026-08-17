#!/usr/bin/env python3
"""
F1 ACT 同步推理脚本（ROS2 版本）
================================
改编自 g2_ACT_sync.py，将原来基于 agibot_gdk SDK 的推理控制代码，
改为通过 ROS2 topic 与 F1 机器人通讯的推理控制代码。

依据《F1系统接口清单【外发版本】.pdf》，F1 通过以下 ROS2 接口进行控制/观测：
  观测（发布者：系统 → 具身，本脚本订阅）：
    /hal/joint_states                      sensor_msgs/msg/JointState  (单位: 度)
    /motion_ctl/gripper/left/state         sensor_msgs/msg/JointState  (0~100, 100=闭合)
    /motion_ctl/gripper/right/state        sensor_msgs/msg/JointState  (0~100, 100=闭合)
    /camera/head/color/image_raw           sensor_msgs/msg/Image
    /camera/left_wrist/color/image_raw     sensor_msgs/msg/Image
    /camera/right_wrist/color/image_raw    sensor_msgs/msg/Image

  控制（接入方式：订阅，本脚本发布）：
    /motion_ctl/joint_ctl                  sensor_msgs/msg/JointState  (单位: 度，需传入整条手臂7个关节)
    /motion_ctl/gripper/left               control_msgs/msg/GripperCommand (0~100)
    /motion_ctl/gripper/right              control_msgs/msg/GripperCommand (0~100)

与 g2_ACT_sync.py 的关键差异（依据本次任务提供的信息）：
  1. 训练数据 action 间隔为 30Hz，因此不再像 G2 那样把动作插值到 100Hz 伺服节拍，
     而是直接按照 30Hz（dt≈33.3ms）逐帧下发 chunk 中的每个 action。
  2. F1 走 ros2 topic 通讯，控制器在未收到新指令前会保持最后一次下发的位置，
     因此推理期间（同步调用模型前向）不需要像 G2 那样起一个线程持续发送 hold 指令，
     直接同步推理即可，机械臂会自然停在上一条指令位置。
  3. 训练数据中的图像是中心裁剪后的 16:9 RGB 图像，IMAGE_HEIGHT=360, IMAGE_WIDTH=640，
     因此从相机原始分辨率获取图像后，需要先按 16:9 做中心裁剪，再 resize 到 640x360。

单位换算说明：
  - ACT 模型 qpos/action 的手臂 14 维沿用 G2 训练时的弧度(rad)表示，
    与 F1 接口文档给出的角度限位（度）在数值上一一对应（如 J1: ±178° ≈ ±3.061rad）。
  - F1 夹爪 topic 使用 [0, 100] 的百分比（100=闭合，0=张开）。

用法:
  source /opt/ros/<distro>/setup.bash
  python f1_ACT_sync.py \
      --ckpt_dir ./f1_ready_86 \
      --task_name f1_ready_86 \
      --max_chunks 40 2>&1 | tee experiment.log
"""

import os, sys, time, argparse, pickle, csv, threading
import numpy as np
import cv2

if '--gpu' in sys.argv:
    idx = sys.argv.index('--gpu')
    if idx + 1 < len(sys.argv):
        os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[idx + 1]

import torch
torch.backends.cudnn.enabled = False  # Thor (aarch64) cuDNN 兼容性问题

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import JointState, Image
from control_msgs.msg import GripperCommand

from policy import ACTPolicy
from constants import REAL_TASK_CONFIGS

# ── 常量 ──────────────────────────────────────────────────────
# 训练数据 action 间隔为 30Hz，直接按该节拍下发，无需插值到高频伺服
ACTION_HZ = 30.0
ACTION_DT = 1.0 / ACTION_HZ

# 训练图像目标尺寸：中心裁剪 16:9 后 resize
IMAGE_HEIGHT = 360
IMAGE_WIDTH = 640

# F1 topic 名称
JOINT_STATES_TOPIC = '/hal/joint_states'
JOINT_CTL_TOPIC = '/motion_ctl/joint_ctl'
GRIPPER_STATE_TOPIC = {'left': '/motion_ctl/gripper/left/state',
                        'right': '/motion_ctl/gripper/right/state'}
GRIPPER_CTL_TOPIC = {'left': '/motion_ctl/gripper/left',
                      'right': '/motion_ctl/gripper/right'}

CAMERA_TOPICS = {
    'hand_left_color':  '/camera/left_wrist/color/image_raw',
    'hand_right_color': '/camera/right_wrist/color/image_raw',
    'head_color':       '/camera/head/color/image_raw',
}

ARM_JOINT_NAMES = (
    ['arm_l_j1', 'arm_l_j2', 'arm_l_j3', 'arm_l_j4', 'arm_l_j5', 'arm_l_j6', 'arm_l_j7'] +
    ['arm_r_j1', 'arm_r_j2', 'arm_r_j3', 'arm_r_j4', 'arm_r_j5', 'arm_r_j6', 'arm_r_j7']
)

GRIPPER_RANGE = (0.0, 100.0)  # F1 夹爪百分比: 0=完全张开, 100=闭合

# F1 手臂关节限位（接口文档，单位: 度），转换为弧度并留 1° 安全裕度
_ARM_LIMIT_LOW_DEG = np.array([-177, -119, -177, -144, -177, -59, -89] * 2, dtype=np.float32)
_ARM_LIMIT_HIGH_DEG = np.array([177, 119, 177, 59, 177, 59, 89] * 2, dtype=np.float32)
ARM_JOINT_LIMITS_LOW = np.deg2rad(_ARM_LIMIT_LOW_DEG)
ARM_JOINT_LIMITS_HIGH = np.deg2rad(_ARM_LIMIT_HIGH_DEG)
MAX_JOINT_DELTA = np.deg2rad(15.0)  # 每个 30Hz 控制周期允许的最大关节角变化(rad)

SENSOR_TIMEOUT = 0.5  # 观测数据陈旧判定阈值(s)


# ── 辅助函数 ──────────────────────────────────────────────────

def load_policy(ckpt_dir, ckpt_name, policy_config, device):
    policy = ACTPolicy(policy_config)
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy.deserialize(torch.load(ckpt_path, map_location=device))
    policy.to(device).eval()
    print(f"✅ 策略加载: {ckpt_path}")
    return policy


def load_stats(ckpt_dir, device):
    with open(os.path.join(ckpt_dir, 'dataset_stats.pkl'), 'rb') as f:
        stats = pickle.load(f)
    for k in stats:
        stats[k] = torch.from_numpy(stats[k]).float().to(device)
    return stats


def center_crop_16_9_and_resize(img, out_w=IMAGE_WIDTH, out_h=IMAGE_HEIGHT):
    """把任意分辨率的图像按 16:9 中心裁剪，再 resize 到 (out_w, out_h)。"""
    h, w = img.shape[:2]
    target_ratio = out_w / out_h  # 16/9
    cur_ratio = w / h
    if cur_ratio > target_ratio:
        # 太宽，裁剪左右
        new_w = int(round(h * target_ratio))
        x0 = (w - new_w) // 2
        img = img[:, x0:x0 + new_w]
    elif cur_ratio < target_ratio:
        # 太高，裁剪上下
        new_h = int(round(w / target_ratio))
        y0 = (h - new_h) // 2
        img = img[y0:y0 + new_h, :]
    return cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)


def decode_ros_image(msg):
    """解析 sensor_msgs/msg/Image，返回 RGB numpy 数组。"""
    encoding = msg.encoding.lower()
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    if encoding in ('rgb8', 'bgr8'):
        img = arr.reshape(msg.height, msg.width, 3)
        if encoding == 'bgr8':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    if encoding in ('mono8',):
        img = arr.reshape(msg.height, msg.width)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    raise ValueError(f"不支持的图像编码: {msg.encoding}")


def clip_gripper(v):
    return float(np.clip(v, GRIPPER_RANGE[0], GRIPPER_RANGE[1]))


def safety_clamp(action, prev=None):
    a = action.copy()
    a[:14] = np.clip(a[:14], ARM_JOINT_LIMITS_LOW, ARM_JOINT_LIMITS_HIGH)
    a[14] = clip_gripper(a[14])
    a[15] = clip_gripper(a[15])
    if prev is not None:
        delta = np.clip(a[:14] - prev[:14], -MAX_JOINT_DELTA, MAX_JOINT_DELTA)
        a[:14] = prev[:14] + delta
    return a


# ── ROS2 接口节点 ─────────────────────────────────────────────

class F1RosInterface(Node):
    """封装 F1 机器人的观测订阅与动作发布。"""

    def __init__(self, camera_names):
        super().__init__('f1_act_sync')
        self.camera_names = camera_names
        self._lock = threading.Lock()

        self._joint_positions = None  # dict: name -> 度
        self._joint_stamp = 0.0
        self._grippers = {'left': None, 'right': None}
        self._gripper_stamp = {'left': 0.0, 'right': 0.0}
        self._images = {}
        self._image_stamp = {}

        qos_state = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.create_subscription(JointState, JOINT_STATES_TOPIC, self._on_joint_states, qos_state)
        self.create_subscription(JointState, GRIPPER_STATE_TOPIC['left'],
                                  lambda m: self._on_gripper('left', m), qos_state)
        self.create_subscription(JointState, GRIPPER_STATE_TOPIC['right'],
                                  lambda m: self._on_gripper('right', m), qos_state)
        for name in camera_names:
            self.create_subscription(Image, CAMERA_TOPICS[name],
                                      lambda m, n=name: self._on_image(n, m), qos_state)

        self.joint_ctl_pub = self.create_publisher(JointState, JOINT_CTL_TOPIC, 10)
        self.gripper_pub = {
            side: self.create_publisher(GripperCommand, GRIPPER_CTL_TOPIC[side], 10)
            for side in ('left', 'right')
        }

    @staticmethod
    def _now(msg):
        stamp = msg.header.stamp
        t = float(stamp.sec) + float(stamp.nanosec) / 1e9
        return t if t > 0 else time.time()

    def _on_joint_states(self, msg):
        with self._lock:
            self._joint_positions = dict(zip(msg.name, msg.position))
            self._joint_stamp = self._now(msg)

    def _on_gripper(self, side, msg):
        if not msg.position:
            return
        with self._lock:
            self._grippers[side] = float(msg.position[0])
            self._gripper_stamp[side] = self._now(msg)

    def _on_image(self, name, msg):
        try:
            rgb = decode_ros_image(msg)
        except ValueError as ex:
            self.get_logger().error(str(ex))
            return
        rgb = center_crop_16_9_and_resize(rgb)
        with self._lock:
            self._images[name] = rgb
            self._image_stamp[name] = self._now(msg)

    def is_ready(self):
        with self._lock:
            if self._joint_positions is None:
                return False
            if any(v is None for v in self._grippers.values()):
                return False
            if any(n not in self._images for n in self.camera_names):
                return False
        return True

    def get_qpos(self):
        """返回16维: 14臂关节(rad) + 左右夹爪(0~100)。"""
        with self._lock:
            if self._joint_positions is None:
                raise RuntimeError("尚未收到 /hal/joint_states")
            if time.time() - self._joint_stamp > SENSOR_TIMEOUT:
                raise RuntimeError("joint_states 数据过期")
            missing = [n for n in ARM_JOINT_NAMES if n not in self._joint_positions]
            if missing:
                raise RuntimeError(f"joint_states 缺少关节: {missing}")
            arm_deg = np.array([self._joint_positions[n] for n in ARM_JOINT_NAMES], dtype=np.float32)
            grippers = dict(self._grippers)
            gripper_stamp = dict(self._gripper_stamp)
        for side, stamp in gripper_stamp.items():
            if time.time() - stamp > SENSOR_TIMEOUT:
                raise RuntimeError(f"{side} 夹爪状态过期")
        qpos = np.zeros(16, dtype=np.float32)
        qpos[:14] = np.deg2rad(arm_deg)
        qpos[14] = grippers['left']
        qpos[15] = grippers['right']
        return qpos

    def get_images(self):
        with self._lock:
            for name in self.camera_names:
                if name not in self._images:
                    raise RuntimeError(f"获取 {name} 图像失败")
                if time.time() - self._image_stamp[name] > SENSOR_TIMEOUT:
                    raise RuntimeError(f"{name} 图像数据过期")
            return {n: self._images[n].copy() for n in self.camera_names}

    def send_action(self, action_16, send_gripper=True):
        arm_msg = JointState()
        arm_msg.header.stamp = self.get_clock().now().to_msg()
        arm_msg.header.frame_id = 'base_link'
        arm_msg.name = ARM_JOINT_NAMES
        arm_msg.position = np.rad2deg(action_16[:14]).astype(float).tolist()
        self.joint_ctl_pub.publish(arm_msg)

        if send_gripper:
            left_cmd = GripperCommand()
            left_cmd.position = clip_gripper(action_16[14])
            left_cmd.max_effort = 10.0
            right_cmd = GripperCommand()
            right_cmd.position = clip_gripper(action_16[15])
            right_cmd.max_effort = 10.0
            self.gripper_pub['left'].publish(left_cmd)
            self.gripper_pub['right'].publish(right_cmd)


# ── 推理 ──────────────────────────────────────────────────────

def infer_one_chunk(policy, ros_iface, camera_names, stats, device):
    """
    获取观测 → 推理 → 反归一化。
    要点2: F1 走 ros2 topic 控制，控制器在无新指令时保持最后位置，
    因此本函数同步执行即可，调用方无需在推理期间持续下发 hold 指令。
    返回 (chunk_np, timing_dict)。观测/推理失败时向上抛出异常，由调用方处理。
    """
    timing = {}

    t0 = time.time()
    images = ros_iface.get_images()
    timing['image_ms'] = (time.time() - t0) * 1000

    t0 = time.time()
    qpos = ros_iface.get_qpos()
    timing['qpos_ms'] = (time.time() - t0) * 1000

    t0 = time.time()
    imgs = np.stack([np.transpose(images[n], (2, 0, 1)) for n in camera_names])
    img_t = torch.from_numpy(imgs / 255.0).float().to(device).unsqueeze(0)
    qpos_t = (torch.from_numpy(qpos).float().to(device).unsqueeze(0)
              - stats['qpos_mean']) / stats['qpos_std']
    timing['preprocess_ms'] = (time.time() - t0) * 1000

    t0 = time.time()
    with torch.inference_mode():
        a_hat = policy(qpos_t, img_t)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    timing['model_ms'] = (time.time() - t0) * 1000

    t0 = time.time()
    a_hat = a_hat * stats['action_std'] + stats['action_mean']
    chunk_np = a_hat[0].cpu().numpy()
    timing['postprocess_ms'] = (time.time() - t0) * 1000

    timing['total_ms'] = sum(timing.values())
    return chunk_np, timing


# ── 主函数 ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='F1 ACT 同步推理（ROS2）')
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--ckpt_name', type=str, default='policy_best.ckpt')
    parser.add_argument('--task_name', type=str, default='f1_ready_86')
    parser.add_argument('--max_chunks', type=int, default=40)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--action_hz', type=float, default=ACTION_HZ,
                         help='训练数据 action 频率，直接按此节拍下发，默认30Hz')
    args = parser.parse_args()

    action_dt = 1.0 / args.action_hz

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    task_config = REAL_TASK_CONFIGS[args.task_name]
    camera_names = task_config['camera_names']

    # 加载配置
    with open(os.path.join(args.ckpt_dir, 'config.pkl'), 'rb') as f:
        policy_config = pickle.load(f)['policy_config']
    chunk_size = policy_config['num_queries']

    print(f"📋 chunk_size={chunk_size}, action_hz={args.action_hz}, "
          f"每chunk执行 {chunk_size - 1} 步（跳过chunk[0]），"
          f"动作执行耗时约 {(chunk_size - 1) * action_dt:.2f}s（不含推理耗时），"
          f"推理期间无需hold")

    # 加载模型
    policy = load_policy(args.ckpt_dir, args.ckpt_name, policy_config, device)
    stats = load_stats(args.ckpt_dir, device)

    # 初始化 ROS2
    print("🤖 初始化 ROS2...")
    rclpy.init()
    ros_iface = F1RosInterface(camera_names)
    spin_thread = threading.Thread(target=rclpy.spin, args=(ros_iface,), daemon=True)
    spin_thread.start()

    print("📷 等待传感器数据就绪...")
    deadline = time.time() + 15.0
    while not ros_iface.is_ready():
        if time.time() > deadline:
            rclpy.shutdown()
            raise RuntimeError("等待 F1 传感器数据超时（关节/夹爪/相机）")
        time.sleep(0.1)

    test_img = ros_iface.get_images()
    test_qpos = ros_iface.get_qpos()
    for n, img in test_img.items():
        print(f"   {n}: {img.shape}")
    print(f"   qpos: arm=[{test_qpos[:14].min():.3f},{test_qpos[:14].max():.3f}] "
          f"grip=[{test_qpos[14]:.1f},{test_qpos[15]:.1f}]")

    # GPU 预热
    print("🔥 GPU预热...")
    img_t = torch.from_numpy(
        np.stack([np.transpose(test_img[n], (2, 0, 1)) for n in camera_names]) / 255.0
    ).float().to(device).unsqueeze(0)
    qpos_t = (torch.from_numpy(test_qpos).float().to(device).unsqueeze(0)
              - stats['qpos_mean']) / stats['qpos_std']
    with torch.inference_mode():
        for _ in range(10):
            policy(qpos_t, img_t)
    print("✅ 就绪")

    input("\n✋ 按 Enter 开始推理控制...\n")

    # ── 主循环 ─────────────────────────────────────────────
    prev_action = ros_iface.get_qpos()
    print(f"   起始qpos(完整): {repr(prev_action)}")
    total_steps = 0
    infer_timings = []
    t_start = time.time()

    # ── chunk数值日志 ───────────────────────────────────────
    log_csv_path = os.path.join(args.ckpt_dir, f"chunk_log_{int(t_start)}.csv")
    arm_cols  = ([f"chunk_arm_start_{k}" for k in range(14)] +
                 [f"chunk_arm_end_{k}" for k in range(14)])
    grip_cols = ["chunk_grip_l_start", "chunk_grip_r_start",
                 "chunk_grip_l_end",   "chunk_grip_r_end"]
    gap_cols  = [f"gap_arm_{k}" for k in range(14)] + ["gap_grip_l", "gap_grip_r"]

    current_chunk = None
    prev_chunk = None  # 用于记录上一个chunk，以计算chunk边界跳变量

    with open(log_csv_path, 'w', newline='') as _csv_file:
        _csv_writer = csv.writer(_csv_file)
        _csv_writer.writerow(["ci", "t_start_s", "infer_ms",
                              "exec_steps"] + arm_cols + grip_cols + gap_cols)
        print(f"  📄 chunk日志写入: {log_csv_path}")

        try:
            for ci in range(args.max_chunks):

                # ── 1. 同步推理 ──────────────────────────────────
                # 要点2: F1 通过 ros2 topic 通讯，控制器在未收到新指令前保持最后
                # 一次下发的位置，因此这里直接同步推理，不需要额外线程持续 hold。
                infer_start = time.time()
                current_chunk, timing = infer_one_chunk(
                    policy, ros_iface, camera_names, stats, device)
                infer_ms = (time.time() - infer_start) * 1000

                infer_timings.append(timing)

                # ── 2. 按30Hz直接执行当前chunk（i=1..n-1，跳过chunk[0]）──
                # 跳过 chunk[0] 的原因：
                #   推理期间机器人保持上一条指令值 prev_action，但 get_qpos 读取的是
                #   实际电机值。受重力/弹性影响，两者存在微小偏差，导致 chunk[0] ≠
                #   prev_action。从 i=1 直接执行 prev_action→chunk[1]，避免此偏差。
                # 要点1: 训练数据 action 间隔为30Hz，chunk中相邻元素本身已按30Hz采样，
                #   因此直接以 action_dt 节拍逐帧下发，不再做100Hz插值。
                n = len(current_chunk)
                chunk_steps = 0

                for i in range(1, n):
                    step_start = time.time()
                    raw_action = current_chunk[i]
                    action = safety_clamp(raw_action, prev_action)
                    ros_iface.send_action(action, send_gripper=True)
                    prev_action = action.copy()
                    chunk_steps += 1
                    total_steps += 1

                    elapsed = time.time() - step_start
                    if elapsed < action_dt:
                        time.sleep(action_dt - elapsed)

                # ── 3. 日志 ──────────────────────────────────────
                chunk_arm_start    = current_chunk[0, :14]
                chunk_arm_end      = current_chunk[-1, :14]
                chunk_grip_l_start = current_chunk[0, 14]
                chunk_grip_r_start = current_chunk[0, 15]
                chunk_grip_l_end   = current_chunk[-1, 14]
                chunk_grip_r_end   = current_chunk[-1, 15]

                # gap: 本chunk[0] 与上一chunk[-1] 的跳变量（chunk边界不连续性）
                if prev_chunk is not None:
                    gap_arm  = current_chunk[0, :14] - prev_chunk[-1, :14]
                    gap_grip = np.array([current_chunk[0, 14] - prev_chunk[-1, 14],
                                          current_chunk[0, 15] - prev_chunk[-1, 15]])
                else:
                    gap_arm  = np.zeros(14)
                    gap_grip = np.zeros(2)

                max_gap_deg = np.rad2deg(np.abs(gap_arm).max())
                print(f"[Chunk {ci:3d}] t={time.time()-t_start:6.1f}s "
                      f"exec={chunk_steps}步 "
                      f"infer={timing['total_ms']:.0f}ms")
                print(f"           arm_start(deg): [{', '.join(f'{np.rad2deg(v):.2f}' for v in chunk_arm_start)}]")
                print(f"           arm_end(deg):   [{', '.join(f'{np.rad2deg(v):.2f}' for v in chunk_arm_end)}]")
                print(f"           grip: L {chunk_grip_l_start:.1f}->{chunk_grip_l_end:.1f}  "
                      f"R {chunk_grip_r_start:.1f}->{chunk_grip_r_end:.1f}")
                print(f"           gap←prev:  arm_max={max_gap_deg:.2f}deg  "
                      f"grip_L={gap_grip[0]:+.1f} grip_R={gap_grip[1]:+.1f}"
                      + ("  ⚠️ 大跳变!" if max_gap_deg > 8.0 else ""))

                _csv_writer.writerow(
                    [ci, f"{time.time()-t_start:.3f}",
                     f"{timing['total_ms']:.1f}", chunk_steps] +
                    [f"{v:.6f}" for v in chunk_arm_start] +
                    [f"{v:.6f}" for v in chunk_arm_end] +
                    [f"{chunk_grip_l_start:.6f}", f"{chunk_grip_r_start:.6f}",
                     f"{chunk_grip_l_end:.6f}",   f"{chunk_grip_r_end:.6f}"] +
                    [f"{v:.6f}" for v in gap_arm] +
                    [f"{gap_grip[0]:.6f}", f"{gap_grip[1]:.6f}"]
                )
                _csv_file.flush()

                prev_chunk = current_chunk  # 保存供下一轮计算gap

        except Exception as _ex:
            print(f"\n⚠️ 异常退出: {_ex}")
        finally:
            # 保持当前位置（仅发送一次手臂指令，无需循环hold，
            # 控制器会在没有新指令时保持该位置）
            if prev_action is not None:
                try:
                    ros_iface.send_action(prev_action, send_gripper=False)
                except Exception:
                    pass
            print(f"  📄 chunk日志已保存: {log_csv_path}")

            total_time = time.time() - t_start
            if total_steps > 0:
                print(f"\n✅ 完成: {total_steps}步 ({len(infer_timings)} chunks), "
                      f"{total_time:.1f}s, {total_steps/total_time:.1f}Hz")

            if infer_timings:
                print(f"\n📊 推理计时 ({len(infer_timings)}次):")
                for key in ['image_ms', 'qpos_ms', 'preprocess_ms', 'model_ms',
                            'postprocess_ms', 'total_ms']:
                    vals = [t[key] for t in infer_timings]
                    print(f"  {key:16s}: mean={np.mean(vals):6.1f}  std={np.std(vals):5.1f}  "
                          f"min={np.min(vals):6.1f}  max={np.max(vals):6.1f}  "
                          f"p95={np.percentile(vals, 95):6.1f}")

            rclpy.shutdown()
            spin_thread.join(timeout=2)


if __name__ == '__main__':
    main()
