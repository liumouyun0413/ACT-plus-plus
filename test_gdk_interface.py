#!/usr/bin/env python3
"""
GDK 接口测试脚本
=================
在 ACT 推理部署之前，先测试 agibot_gdk 的核心接口：
  1. 相机图像读取（3路：head_color, hand_left_color, hand_right_color）
  2. 关节状态读取（22全身关节 + 夹爪末端状态）
  3. 关节伺服控制（小幅正弦运动测试）
  4. 夹爪控制（omnipicker 开合测试）

使用方法:
  source /home/liumouyun/Downloads/ACT-plus-plus/app/env.sh
  python test_gdk_interface.py [--test all|camera|joint|servo|gripper] [--save_images]
"""

import os
import sys
import time
import math
import argparse
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("⚠️ OpenCV 未安装，图像保存功能不可用")

import agibot_gdk

# ═══════════════════════════════════════════════════════════════
# 常量
# ═══════════════════════════════════════════════════════════════

CAMERA_LIST = [
    ('head_color',       agibot_gdk.CameraType.kHeadColor,       "头部彩色相机"),
    ('hand_left_color',  agibot_gdk.CameraType.kHandLeftColor,   "左手彩色相机"),
    ('hand_right_color', agibot_gdk.CameraType.kHandRightColor,  "右手彩色相机"),
]

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
ARM_JOINT_INDICES = list(range(8, 22))  # 14个手臂关节

GRIPPER_RANGE = (-0.785, 0.0)  # omnipicker 范围


# ═══════════════════════════════════════════════════════════════
# 图像解码
# ═══════════════════════════════════════════════════════════════

def decode_image(image_obj):
    """GDK image → numpy BGR (H,W,3)"""
    if image_obj is None:
        return None
    if image_obj.encoding in (agibot_gdk.Encoding.JPEG, agibot_gdk.Encoding.PNG):
        nparr = np.frombuffer(image_obj.data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif image_obj.encoding == agibot_gdk.Encoding.UNCOMPRESSED:
        img = np.frombuffer(image_obj.data, dtype=np.uint8).reshape(
            (image_obj.height, image_obj.width, 3))
        if image_obj.color_format == agibot_gdk.ColorFormat.RGB:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    return None


# ═══════════════════════════════════════════════════════════════
# 测试 1: 相机
# ═══════════════════════════════════════════════════════════════

def test_camera(camera, save_images=False):
    print("\n" + "=" * 60)
    print("📷 测试相机图像读取")
    print("=" * 60)

    save_dir = "test_gdk_images"
    if save_images:
        os.makedirs(save_dir, exist_ok=True)

    all_ok = True
    for cam_name, cam_type, cam_desc in CAMERA_LIST:
        print(f"\n--- {cam_desc} ({cam_name}) ---")
        try:
            img_obj = camera.get_latest_image(cam_type, 2000.0)
            if img_obj is None:
                print(f"  ❌ 返回 None")
                all_ok = False
                continue

            print(f"  时间戳:   {img_obj.timestamp_ns}")
            print(f"  尺寸:     {img_obj.width} x {img_obj.height}")
            print(f"  编码:     {img_obj.encoding}")
            print(f"  颜色格式: {img_obj.color_format}")

            if HAS_CV2:
                bgr = decode_image(img_obj)
                if bgr is not None:
                    print(f"  解码后:   shape={bgr.shape}, dtype={bgr.dtype}")
                    if save_images:
                        path = os.path.join(save_dir, f"{cam_name}.jpg")
                        cv2.imwrite(path, bgr)
                        print(f"  💾 已保存: {path}")
                else:
                    print(f"  ❌ 解码失败")
                    all_ok = False

            # 测试帧率
            try:
                fps = camera.get_image_fps(cam_type)
                print(f"  帧率:     {fps:.1f} FPS")
            except Exception as e:
                print(f"  帧率获取失败: {e}")

        except Exception as e:
            print(f"  ❌ 异常: {e}")
            all_ok = False

    # 连续读取测试延迟
    print(f"\n--- 连续读取 10 帧延迟测试 (head_color) ---")
    latencies = []
    for i in range(10):
        t0 = time.time()
        img_obj = camera.get_latest_image(agibot_gdk.CameraType.kHeadColor, 1000.0)
        dt = (time.time() - t0) * 1000
        latencies.append(dt)
    latencies = np.array(latencies)
    print(f"  读取延迟: mean={latencies.mean():.1f}ms, "
          f"min={latencies.min():.1f}ms, max={latencies.max():.1f}ms")

    # 三路同时读取
    print(f"\n--- 三路相机同时读取延迟 ---")
    t0 = time.time()
    for cam_name, cam_type, _ in CAMERA_LIST:
        camera.get_latest_image(cam_type, 1000.0)
    dt = (time.time() - t0) * 1000
    print(f"  三路总延迟: {dt:.1f}ms")

    status = "✅ 全部通过" if all_ok else "⚠️ 部分失败"
    print(f"\n相机测试结果: {status}")
    return all_ok


# ═══════════════════════════════════════════════════════════════
# 测试 2: 关节状态
# ═══════════════════════════════════════════════════════════════

def test_joint_states(robot):
    print("\n" + "=" * 60)
    print("🦾 测试关节状态读取")
    print("=" * 60)

    all_ok = True

    # --- 全身关节 ---
    print("\n--- get_joint_states() ---")
    try:
        js = robot.get_joint_states()
        print(f"  关节数量: {js['nums']}")
        print(f"  时间戳:   {js['timestamp']}")
        print(f"\n  {'序号':<4} {'名称':<28} {'位置(rad)':>10} {'速度':>10} {'力矩':>10}")
        print(f"  {'─'*4} {'─'*28} {'─'*10} {'─'*10} {'─'*10}")
        for i, s in enumerate(js['states']):
            marker = " ◄ARM" if i in ARM_JOINT_INDICES else ""
            print(f"  {i:<4} {s['name']:<28} {s['position']:>10.4f} "
                  f"{s['velocity']:>10.4f} {s['effort']:>10.4f}{marker}")
    except Exception as e:
        print(f"  ❌ 异常: {e}")
        all_ok = False

    # --- 提取 ACT 用的 16 维 qpos ---
    print(f"\n--- 提取 ACT 16-dim qpos ---")
    try:
        all_positions = [s['motor_position'] for s in js['states']]
        arm_pos = np.array([all_positions[i] for i in ARM_JOINT_INDICES], dtype=np.float32)
        print(f"  14臂关节: {arm_pos}")

        end_state = robot.get_end_state()
        left_g = end_state['left_end_state']['end_states'][0]['position']
        right_g = end_state['right_end_state']['end_states'][0]['position']
        print(f"  左夹爪:   {left_g:.4f}")
        print(f"  右夹爪:   {right_g:.4f}")

        qpos = np.zeros(16, dtype=np.float32)
        qpos[:14] = arm_pos
        qpos[14] = left_g
        qpos[15] = right_g
        print(f"\n  qpos (16-dim): {qpos}")
        print(f"  臂关节范围: [{arm_pos.min():.4f}, {arm_pos.max():.4f}]")
    except Exception as e:
        print(f"  ❌ 异常: {e}")
        all_ok = False

    # --- 末端执行器详情 ---
    print(f"\n--- get_end_state() 详情 ---")
    try:
        end_state = robot.get_end_state()
        for side in ['left', 'right']:
            es = end_state[f'{side}_end_state']
            print(f"  {side} 执行器:")
            print(f"    类型:     {es['type']}")
            print(f"    受控:     {es['controlled']}")
            print(f"    关节名:   {es['names']}")
            if es['end_states']:
                for j, st in enumerate(es['end_states']):
                    print(f"    关节{j}: pos={st['position']:.4f}, "
                          f"vel={st['velocity']:.4f}, cur={st['current']:.4f}")
    except Exception as e:
        print(f"  ❌ 异常: {e}")
        all_ok = False

    # --- 连续读取频率测试 ---
    print(f"\n--- 关节状态读取频率测试 (100次) ---")
    times = []
    for _ in range(100):
        t0 = time.time()
        robot.get_joint_states()
        times.append((time.time() - t0) * 1000)
    times = np.array(times)
    print(f"  单次延迟: mean={times.mean():.2f}ms, "
          f"min={times.min():.2f}ms, max={times.max():.2f}ms")
    print(f"  等效频率: {1000/times.mean():.0f} Hz")

    status = "✅ 全部通过" if all_ok else "⚠️ 部分失败"
    print(f"\n关节状态测试结果: {status}")
    return all_ok


# ═══════════════════════════════════════════════════════════════
# 测试 3: 手臂伺服控制 (小幅正弦抖动)
# ═══════════════════════════════════════════════════════════════

def test_servo_control(robot, duration=5.0, amplitude_deg=2.0):
    print("\n" + "=" * 60)
    print("🎮 测试手臂伺服控制 (小幅正弦运动)")
    print("=" * 60)
    print(f"  幅度: {amplitude_deg}° ({math.radians(amplitude_deg):.4f} rad)")
    print(f"  持续: {duration}s")
    print(f"  频率: 0.5 Hz (只在最后一个关节上运动)")

    # 读取当前位置作为基线
    js = robot.get_joint_states()
    all_pos = [s['motor_position'] for s in js['states']]
    arm_baseline = [all_pos[i] for i in ARM_JOINT_INDICES]
    print(f"  当前臂关节基线: {[f'{p:.3f}' for p in arm_baseline]}")

    input("\n  ⚠️ 将对最后一个右臂关节做小幅运动测试，确认安全后按 Enter 继续...")

    amp = math.radians(amplitude_deg)
    freq = 0.5
    omega = 2 * math.pi * freq
    dt = 0.01  # 100Hz
    t = 0.0
    fade_duration = 1.0
    count = 0

    print(f"  🚀 开始运动... (Ctrl+C 停止)")
    try:
        while t < duration:
            fade = min(1.0, t / fade_duration)
            arm_cmd = arm_baseline.copy()
            # 只在最后一个关节 (idx=13, 即 idx67_arm_r_joint7) 上做正弦
            arm_cmd[13] = arm_baseline[13] + fade * amp * math.sin(omega * t)

            robot.servo_control_arm_pos(arm_cmd, 2)

            count += 1
            if count % 100 == 0:
                print(f"    t={t:.1f}s, joint13_cmd={arm_cmd[13]:.4f} "
                      f"(base={arm_baseline[13]:.4f}, delta={arm_cmd[13]-arm_baseline[13]:.4f})")

            t += dt
            time.sleep(dt)

    except KeyboardInterrupt:
        print("\n  ⏹ 用户中断")

    # 恢复到基线位置
    print(f"  🔄 恢复基线位置...")
    for _ in range(50):
        robot.servo_control_arm_pos(arm_baseline, 2)
        time.sleep(0.01)

    print(f"  ✅ 伺服控制测试完成, 发送了 {count} 次指令")
    return True


# ═══════════════════════════════════════════════════════════════
# 测试 4: 夹爪控制
# ═══════════════════════════════════════════════════════════════

def test_gripper_control(robot):
    print("\n" + "=" * 60)
    print("🤏 测试夹爪控制 (omnipicker)")
    print("=" * 60)
    print(f"  范围: [{GRIPPER_RANGE[0]}, {GRIPPER_RANGE[1]}] rad")
    print(f"  -0.785 = 全开, 0.0 = 全闭")

    # 读取当前夹爪位置
    end_state = robot.get_end_state()
    left_cur = end_state['left_end_state']['end_states'][0]['position']
    right_cur = end_state['right_end_state']['end_states'][0]['position']
    print(f"  当前位置: 左={left_cur:.4f}, 右={right_cur:.4f}")

    input("\n  ⚠️ 将控制左夹爪开合测试，确认安全后按 Enter 继续...")

    def send_left_gripper(pos):
        js = agibot_gdk.JointStates()
        js.group = "left_tool"
        js.target_type = "omnipicker"
        st = agibot_gdk.JointState()
        st.position = float(np.clip(pos, GRIPPER_RANGE[0], GRIPPER_RANGE[1]))
        js.states = [st]
        js.nums = 1
        robot.move_ee_pos(js)

    # 打开
    print(f"  → 左夹爪全开 (-0.785)...")
    send_left_gripper(-0.785)
    time.sleep(2)

    end_state = robot.get_end_state()
    pos = end_state['left_end_state']['end_states'][0]['position']
    print(f"    实际位置: {pos:.4f}")

    # 关闭
    print(f"  → 左夹爪全闭 (0.0)...")
    send_left_gripper(0.0)
    time.sleep(2)

    end_state = robot.get_end_state()
    pos = end_state['left_end_state']['end_states'][0]['position']
    print(f"    实际位置: {pos:.4f}")

    # 恢复
    print(f"  → 恢复原始位置 ({left_cur:.4f})...")
    send_left_gripper(left_cur)
    time.sleep(1)

    print(f"  ✅ 夹爪控制测试完成")
    return True


# ═══════════════════════════════════════════════════════════════
# 综合测试: 模拟推理循环的读取时序
# ═══════════════════════════════════════════════════════════════

def test_inference_timing(camera, robot, n_loops=20):
    print("\n" + "=" * 60)
    print("⏱️  模拟推理循环时序测试")
    print("=" * 60)
    print(f"  模拟 {n_loops} 步推理循环 (不含模型推理)")

    cam_types = [ct for _, ct, _ in CAMERA_LIST]
    timings = {'camera': [], 'joint': [], 'total': []}

    for i in range(n_loops):
        t_total = time.time()

        # 读相机
        t0 = time.time()
        for ct in cam_types:
            camera.get_latest_image(ct, 1000.0)
        timings['camera'].append((time.time() - t0) * 1000)

        # 读关节
        t0 = time.time()
        robot.get_joint_states()
        robot.get_end_state()
        timings['joint'].append((time.time() - t0) * 1000)

        timings['total'].append((time.time() - t_total) * 1000)

    for key in timings:
        arr = np.array(timings[key])
        print(f"  {key:>8s}: mean={arr.mean():6.1f}ms, "
              f"min={arr.min():6.1f}ms, max={arr.max():6.1f}ms, "
              f"std={arr.std():5.1f}ms")

    overhead = np.array(timings['total']).mean()
    budget = 230  # 4.3Hz → 230ms
    print(f"\n  控制周期预算: {budget}ms (4.3Hz)")
    print(f"  观测读取开销: {overhead:.1f}ms")
    print(f"  剩余推理预算: {budget - overhead:.1f}ms")
    print(f"  ✅ 时序测试完成")


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='GDK 接口功能测试')
    parser.add_argument('--test', type=str, default='all',
                        choices=['all', 'camera', 'joint', 'servo', 'gripper', 'timing'],
                        help='选择测试项 (default: all)')
    parser.add_argument('--save_images', action='store_true',
                        help='保存相机图像到 test_gdk_images/')
    parser.add_argument('--servo_duration', type=float, default=5.0,
                        help='伺服测试持续时间(秒)')
    parser.add_argument('--servo_amplitude', type=float, default=2.0,
                        help='伺服测试幅度(度)')
    args = parser.parse_args()

    print("🤖 初始化 GDK...")
    agibot_gdk.gdk_init()

    camera = None
    robot = None

    need_camera = args.test in ('all', 'camera', 'timing')
    need_robot = args.test in ('all', 'joint', 'servo', 'gripper', 'timing')

    if need_camera:
        print("  初始化 Camera...")
        camera = agibot_gdk.Camera()

    if need_robot:
        print("  初始化 Robot...")
        robot = agibot_gdk.Robot()

    time.sleep(2)
    print("✅ GDK 初始化完成\n")

    results = {}

    if args.test in ('all', 'camera'):
        results['camera'] = test_camera(camera, save_images=args.save_images)

    if args.test in ('all', 'joint'):
        results['joint'] = test_joint_states(robot)

    if args.test in ('all', 'servo'):
        results['servo'] = test_servo_control(robot, args.servo_duration, args.servo_amplitude)

    if args.test in ('all', 'gripper'):
        results['gripper'] = test_gripper_control(robot)

    if args.test in ('all', 'timing'):
        test_inference_timing(camera, robot)

    # 汇总
    print("\n" + "=" * 60)
    print("📊 测试汇总")
    print("=" * 60)
    for name, ok in results.items():
        icon = "✅" if ok else "❌"
        print(f"  {icon} {name}")
    print()


if __name__ == '__main__':
    main()
