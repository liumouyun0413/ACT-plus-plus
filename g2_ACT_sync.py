#!/usr/bin/env python3
"""
G2 ACT 完全同步推理脚本
========================
逻辑: 执行当前chunk → chunk结束后同步推理下一chunk（期间hold当前位置）→ 循环

无任何预取/异步线程，逻辑最简单，回退最小：
  - 推理时观测点 = chunk末尾真实位置，chunk[0] ≈ qpos_obs
  - 但 hold 期间指令值(prev_action) ≠ 实际电机值(qpos_obs)，执行 chunk[0] 会产生微小回退
  - 因此从 i=1 开始执行（prev_action→chunk[1]），完全跳过 chunk[0]

代价: 每个chunk边界有 ~100ms 的hold停顿

用法:
  source ~/app/env.sh
  python g2_ACT_sync.py \
      --ckpt_dir ./sorting_blocks2 \
      --task_name sorting_blocks \
      --max_chunks 11 2>&1 | tee experiment.log
"""

import os, sys, time, argparse, pickle, csv
import numpy as np
import cv2

if '--gpu' in sys.argv:
    idx = sys.argv.index('--gpu')
    if idx + 1 < len(sys.argv):
        os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[idx + 1]

import torch
torch.backends.cudnn.enabled = False  # Thor (aarch64) cuDNN 兼容性问题
import agibot_gdk
from policy import ACTPolicy
from constants import REAL_TASK_CONFIGS

# ── 常量 ──────────────────────────────────────────────────────
SERVO_DT = 0.01  # 100Hz

CAMERA_MAP = {
    'hand_left_color':  agibot_gdk.CameraType.kHandLeftColor,
    'hand_right_color': agibot_gdk.CameraType.kHandRightColor,
    'head_color':       agibot_gdk.CameraType.kHeadColor,
}
IMG_SIZE = (480, 640)

ALL_JOINT_NAMES = [
    "idx01_body_joint1", "idx02_body_joint2", "idx03_body_joint3",
    "idx04_body_joint4", "idx05_body_joint5",
    "idx11_head_joint1", "idx13_head_joint3", "idx12_head_joint2",
    "idx21_arm_l_joint1", "idx22_arm_l_joint2", "idx23_arm_l_joint3",
    "idx24_arm_l_joint4", "idx25_arm_l_joint5", "idx26_arm_l_joint6",
    "idx27_arm_l_joint7",
    "idx61_arm_r_joint1", "idx62_arm_r_joint2", "idx63_arm_r_joint3",
    "idx64_arm_r_joint4", "idx65_arm_r_joint5", "idx66_arm_r_joint6",
    "idx67_arm_r_joint7",
]
ARM_JOINT_INDICES = list(range(8, 22))
GRIPPER_RANGE = (-0.785, 0.0)  # omnipicker 官方范围

# joint_servo_control 统一控制: 14臂关节 + 2夹爪关节
SERVO_JOINT_NAMES = [
    "idx21_arm_l_joint1", "idx22_arm_l_joint2", "idx23_arm_l_joint3",
    "idx24_arm_l_joint4", "idx25_arm_l_joint5", "idx26_arm_l_joint6",
    "idx27_arm_l_joint7",
    "idx61_arm_r_joint1", "idx62_arm_r_joint2", "idx63_arm_r_joint3",
    "idx64_arm_r_joint4", "idx65_arm_r_joint5", "idx66_arm_r_joint6",
    "idx67_arm_r_joint7",
    "idx31_gripper_l_inner_joint1",  # 左夹爪
    "idx71_gripper_r_inner_joint1",  # 右夹爪
]

# GDK v2.6.3 官方关节限位 (稍微收缩0.01rad作为安全边距)
ARM_JOINT_LIMITS_LOW = np.array([
    -3.061, -2.049, -3.061, -2.485, -3.061, -1.002, -1.525,
    -3.061, -2.049, -3.061, -2.485, -3.061, -1.002, -1.525,
], dtype=np.float32)
ARM_JOINT_LIMITS_HIGH = np.array([
     3.061,  2.049,  3.061,  1.002,  3.061,  1.002,  1.525,
     3.061,  2.049,  3.061,  1.002,  3.061,  1.002,  1.525,
], dtype=np.float32)
MAX_JOINT_DELTA = 0.4

# ── 辅助函数 ──────────────────────────────────────────────────

def load_policy(ckpt_dir, ckpt_name, policy_config, device):
    policy = ACTPolicy(policy_config)
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy.deserialize(torch.load(ckpt_path, map_location=device))
    policy.to(device).eval()
    print(f"✅ 策略加载: {ckpt_path}")
    return policy


def load_stats(ckpt_dir):
    with open(os.path.join(ckpt_dir, 'dataset_stats.pkl'), 'rb') as f:
        stats = pickle.load(f)
    for k in stats:
        stats[k] = torch.from_numpy(stats[k]).float().cuda()
    return stats


def decode_gdk_image(img_obj):
    if img_obj is None:
        return None
    if img_obj.encoding in (agibot_gdk.Encoding.JPEG, agibot_gdk.Encoding.PNG):
        return cv2.imdecode(np.frombuffer(img_obj.data, np.uint8), cv2.IMREAD_COLOR)
    elif img_obj.encoding == agibot_gdk.Encoding.UNCOMPRESSED:
        img = np.frombuffer(img_obj.data, np.uint8).reshape(img_obj.height, img_obj.width, 3)
        if img_obj.color_format == agibot_gdk.ColorFormat.RGB:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    return None


def get_images(camera, camera_names):
    images = {}
    for name in camera_names:
        bgr = decode_gdk_image(camera.get_latest_image(CAMERA_MAP[name], 500.0))
        if bgr is None:
            raise RuntimeError(f"获取 {name} 图像失败")
        bgr = cv2.resize(bgr, (IMG_SIZE[1], IMG_SIZE[0]))
        images[name] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return images


def get_qpos(robot):
    js = robot.get_joint_states()
    positions = [s['motor_position'] for s in js['states']]
    arm = np.array([positions[i] for i in ARM_JOINT_INDICES], dtype=np.float32)
    es = robot.get_end_state()
    lg = es['left_end_state']['end_states'][0]['position']
    rg = es['right_end_state']['end_states'][0]['position']
    qpos = np.zeros(16, dtype=np.float32)
    qpos[:14] = arm
    qpos[14] = lg
    qpos[15] = rg
    return qpos


def clip_gripper(v):
    return float(np.clip(v, GRIPPER_RANGE[0], GRIPPER_RANGE[1]))


def send_action(robot, action_16, send_gripper=True):
    req = agibot_gdk.JointServoControlReq()
    req.control_period = SERVO_DT * 1.5
    if send_gripper:
        req.joint_names = SERVO_JOINT_NAMES
        positions = action_16[:14].tolist() + [
            clip_gripper(action_16[14]),
            clip_gripper(action_16[15]),
        ]
    else:
        req.joint_names = SERVO_JOINT_NAMES[:14]
        positions = action_16[:14].tolist()
    req.joint_positions = positions
    robot.joint_servo_control(req)


def safety_clamp(action, prev=None):
    a = action.copy()
    a[:14] = np.clip(a[:14], ARM_JOINT_LIMITS_LOW, ARM_JOINT_LIMITS_HIGH)
    a[14] = np.clip(a[14], *GRIPPER_RANGE)
    a[15] = np.clip(a[15], *GRIPPER_RANGE)
    if prev is not None:
        delta = np.clip(a[:14] - prev[:14], -MAX_JOINT_DELTA, MAX_JOINT_DELTA)
        a[:14] = prev[:14] + delta
    return a


def infer_one_chunk(policy, camera, robot, camera_names, stats):
    """
    获取观测 → 推理 → 反归一化。
    推理期间调用方应持续 hold，本函数内部不控制机器人。
    返回 (chunk_np, timing_dict)
    """
    timing = {}

    t0 = time.time()
    images = get_images(camera, camera_names)
    timing['image_ms'] = (time.time() - t0) * 1000

    t0 = time.time()
    qpos = get_qpos(robot)
    timing['qpos_ms'] = (time.time() - t0) * 1000

    t0 = time.time()
    imgs = np.stack([np.transpose(images[n], (2, 0, 1)) for n in camera_names])
    img_t = torch.from_numpy(imgs / 255.0).float().cuda().unsqueeze(0)
    qpos_t = (torch.from_numpy(qpos).float().cuda().unsqueeze(0)
              - stats['qpos_mean']) / stats['qpos_std']
    timing['preprocess_ms'] = (time.time() - t0) * 1000

    t0 = time.time()
    with torch.inference_mode():
        a_hat = policy(qpos_t, img_t)
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
    parser = argparse.ArgumentParser(description='G2 ACT 完全同步推理')
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--ckpt_name', type=str, default='policy_best.ckpt')
    parser.add_argument('--task_name', type=str, default='sorting_blocks')
    parser.add_argument('--max_chunks', type=int, default=20)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--interp_steps', type=int, default=23,
                        help='每两个action间插值步数 (100Hz). 训练数据4.3Hz→每步0.233s→23步@100Hz')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    task_config = REAL_TASK_CONFIGS[args.task_name]
    camera_names = task_config['camera_names']

    # 加载配置
    with open(os.path.join(args.ckpt_dir, 'config.pkl'), 'rb') as f:
        policy_config = pickle.load(f)['policy_config']
    chunk_size = policy_config['num_queries']

    actual_interp = (chunk_size - 1) * args.interp_steps  # range(1,n) 共 n-1 段
    chunk_duration = actual_interp * SERVO_DT
    print(f"📋 chunk_size={chunk_size}, interp_steps={args.interp_steps}, "
          f"实际执行 {actual_interp}步@100Hz = {chunk_duration:.2f}s, "
          f"推理期间 hold ~100ms")

    # 加载模型
    policy = load_policy(args.ckpt_dir, args.ckpt_name, policy_config, device)
    stats = load_stats(args.ckpt_dir)

    # 初始化GDK
    print("🤖 初始化 GDK...")
    agibot_gdk.gdk_init()
    camera = agibot_gdk.Camera()
    robot = agibot_gdk.Robot()
    time.sleep(2)

    # 测试读取
    print("📷 测试传感器...")
    test_img = get_images(camera, camera_names)
    test_qpos = get_qpos(robot)
    for n, img in test_img.items():
        print(f"   {n}: {img.shape}")
    print(f"   qpos: arm=[{test_qpos[:14].min():.3f},{test_qpos[:14].max():.3f}] "
          f"grip=[{test_qpos[14]:.3f},{test_qpos[15]:.3f}]")

    # GPU预热
    print("🔥 GPU预热...")
    img_t = torch.from_numpy(
        np.stack([np.transpose(test_img[n], (2, 0, 1)) for n in camera_names]) / 255.0
    ).float().cuda().unsqueeze(0)
    qpos_t = (torch.from_numpy(test_qpos).float().cuda().unsqueeze(0)
              - stats['qpos_mean']) / stats['qpos_std']
    with torch.inference_mode():
        for _ in range(10):
            policy(qpos_t, img_t)
    print("✅ 就绪")

    input("\n✋ 按 Enter 开始推理控制...\n")

    # ── 主循环 ─────────────────────────────────────────────
    prev_action = get_qpos(robot)
    print(f"   起始qpos(完整): {repr(prev_action)}")
    total_steps = 0
    infer_timings = []
    t_start = time.time()

    # ── chunk数值日志 ───────────────────────────────────────
    log_csv_path = os.path.join(args.ckpt_dir, f"chunk_log_{int(t_start)}.csv")
    _csv_file = open(log_csv_path, 'w', newline='')
    _csv_writer = csv.writer(_csv_file)
    arm_cols  = ([f"chunk_arm_start_{k}" for k in range(14)] +
                 [f"chunk_arm_end_{k}" for k in range(14)])
    grip_cols = ["chunk_grip_l_start", "chunk_grip_r_start",
                 "chunk_grip_l_end",   "chunk_grip_r_end"]
    gap_cols  = [f"gap_arm_{k}" for k in range(14)] + ["gap_grip_l", "gap_grip_r"]
    _csv_writer.writerow(["ci", "t_start_s", "hold_steps", "infer_ms",
                          "exec_steps"] + arm_cols + grip_cols + gap_cols)
    print(f"  📄 chunk日志写入: {log_csv_path}")

    current_chunk = None
    prev_chunk = None  # 用于记录上一个chunk，以计算chunk边界跳变量

    try:
        for ci in range(args.max_chunks):

            # ── 1. 同步推理（hold prev_action 直到完成）─────────
            infer_start = time.time()
            hold_count = 0

            # 启动推理线程（为了能在推理期间持续 hold，用线程包一层）
            import threading
            infer_result = {'chunk': None, 'timing': None, 'done': False}

            def _do_infer():
                try:
                    c, t = infer_one_chunk(policy, camera, robot, camera_names, stats)
                    infer_result['chunk'] = c
                    infer_result['timing'] = t
                except Exception as ex:
                    print(f"  ⚠️ 推理异常: {ex}")
                finally:
                    infer_result['done'] = True

            threading.Thread(target=_do_infer, daemon=True).start()

            # 推理期间 hold 当前位置，保持 100Hz 节拍
            while not infer_result['done']:
                step_start = time.time()
                send_action(robot, prev_action, send_gripper=False)
                hold_count += 1
                elapsed = time.time() - step_start
                if elapsed < SERVO_DT:
                    time.sleep(SERVO_DT - elapsed)

            infer_ms = (time.time() - infer_start) * 1000
            current_chunk = infer_result['chunk']
            timing = infer_result['timing']

            if current_chunk is None:
                print("  ⚠️ 推理失败，跳过本chunk")
                continue

            infer_timings.append(timing)

            # ── 2. 插值执行当前chunk（i=1..n-1，跳过chunk[0]）──
            # 跳过 chunk[0] 的原因：
            #   hold 期间发送指令值 prev_action，但 get_qpos 读取的是实际电机值。
            #   受重力/弹性影响，实际值与指令值有微小偏差，导致 chunk[0] ≠ prev_action。
            #   若某关节实际值 < 指令值，i=0 段（prev_action→chunk[0]）会产生短暂回退。
            #   从 i=1 直接执行 prev_action→chunk[1]，完全避免此问题。
            n = len(current_chunk)
            chunk_steps = 0

            for i in range(1, n):
                start = prev_action if i == 1 else current_chunk[i - 1]
                end = current_chunk[i]

                for j in range(1, args.interp_steps + 1):
                    step_start = time.time()
                    alpha = j / args.interp_steps
                    raw_action = (1 - alpha) * start + alpha * end
                    action = safety_clamp(raw_action, prev_action)
                    send_action(robot, action, send_gripper=True)
                    prev_action = action.copy()
                    chunk_steps += 1
                    total_steps += 1

                    elapsed = time.time() - step_start
                    if elapsed < SERVO_DT:
                        time.sleep(SERVO_DT - elapsed)

            # ── 3. 日志 ──────────────────────────────────────
            chunk_arm_start    = current_chunk[0, :14]
            chunk_arm_end      = current_chunk[-1, :14]
            chunk_grip_l_start = current_chunk[0, 14]
            chunk_grip_r_start = current_chunk[0, 15]
            chunk_grip_l_end   = current_chunk[-1, 14]
            chunk_grip_r_end   = current_chunk[-1, 15]

            # gap: 本chunk[0] 与上一chunk[-1] 的跳变量（chunk边界不连续性）
            # ci=0 时无上一chunk，填零
            if prev_chunk is not None:
                gap_arm  = current_chunk[0, :14] - prev_chunk[-1, :14]
                gap_grip = np.array([current_chunk[0, 14] - prev_chunk[-1, 14],
                                     current_chunk[0, 15] - prev_chunk[-1, 15]])
            else:
                gap_arm  = np.zeros(14)
                gap_grip = np.zeros(2)

            max_gap = np.abs(gap_arm).max()
            print(f"[Chunk {ci:3d}] t={time.time()-t_start:6.1f}s "
                  f"hold={hold_count}({infer_ms:.0f}ms) "
                  f"exec={chunk_steps}步 "
                  f"infer={timing['total_ms']:.0f}ms")
            print(f"           arm_start: [{', '.join(f'{v:.4f}' for v in chunk_arm_start)}]")
            print(f"           arm_end:   [{', '.join(f'{v:.4f}' for v in chunk_arm_end)}]")
            print(f"           grip: L {chunk_grip_l_start:.4f}->{chunk_grip_l_end:.4f}  "
                  f"R {chunk_grip_r_start:.4f}->{chunk_grip_r_end:.4f}")
            print(f"           gap←prev:  arm_max={max_gap:.4f}  "
                  f"grip_L={gap_grip[0]:+.4f} grip_R={gap_grip[1]:+.4f}"
                  + ("  ⚠️ 大跳变!" if max_gap > 0.15 else ""))

            _csv_writer.writerow(
                [ci, f"{time.time()-t_start:.3f}", hold_count,
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
        # hold 当前位置 (无论正常结束/异常都执行)
        if prev_action is not None:
            for _ in range(50):
                try:
                    send_action(robot, prev_action, send_gripper=False)
                except Exception:
                    pass
                time.sleep(SERVO_DT)
        _csv_file.close()
        print(f"  📄 chunk日志已保存: {log_csv_path}")

        total_time = time.time() - t_start
        if total_steps > 0:
            print(f"\n✅ 完成: {total_steps}步 ({len(infer_timings)} chunks), "
                  f"{total_time:.1f}s, {total_steps/total_time:.0f}Hz servo")

        if infer_timings:
            print(f"\n📊 推理计时 ({len(infer_timings)}次):")
            for key in ['image_ms', 'qpos_ms', 'preprocess_ms', 'model_ms',
                        'postprocess_ms', 'total_ms']:
                vals = [t[key] for t in infer_timings]
                print(f"  {key:16s}: mean={np.mean(vals):6.1f}  std={np.std(vals):5.1f}  "
                      f"min={np.min(vals):6.1f}  max={np.max(vals):6.1f}  "
                      f"p95={np.percentile(vals, 95):6.1f}")


if __name__ == '__main__':
    main()
