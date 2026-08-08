#!/usr/bin/env python3
"""机器人归位脚本: 使用 GDK move_arm_joint() 规划控制
   GDK 内部做轨迹规划, 阻塞直到到达目标位, 无需手动插值。
"""
import time
import numpy as np

import agibot_gdk

# 训练数据的标准起始位 (16维: 14臂 + 2爪)
HOME_QPOS = np.array([
     1.7300, -1.1500, -1.6000, -1.8000,  1.3300,  0.0000,  0.0000,  # 左臂7
    -1.7300, -1.1500,  1.6000, -1.8000, -1.3300,  0.0000,  0.0000,  # 右臂7
    -0.784,  -0.784,  # 左右夹爪
], dtype=np.float32)

GRIPPER_RANGE = (-0.784, -0.001)
ARM_VELOCITY = 0.1     # move_arm_joint 关节速度 (rad/s), 约5.7°/s
LIFT_VELOCITY = 0.08   # 抬臂阶段稍快一点
ARM_JOINT_INDICES = list(range(8, 22))

# 安全抬臂位: joint1(肩) + joint4(肘) 向上抬
# 实测: 左j1减小=上, 右j1增大=上, j4减小=上(双臂)
LIFT_QPOS = np.array([
     1.4300, -1.1500, -1.6000, -2.1000,  1.3300,  0.0000,  0.0000,  # 左臂: j1-0.3(上), j4-0.3(上)
    -1.4300, -1.1500,  1.6000, -2.1000, -1.3300,  0.0000,  0.0000,  # 右臂: j1+0.3(上), j4-0.3(上)
], dtype=np.float32)


def clip_gripper(v):
    return float(np.clip(v, GRIPPER_RANGE[0], GRIPPER_RANGE[1]))


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


def send_gripper(robot, target):
    """发送夹爪目标位置"""
    for idx, group in [(14, "left_tool"), (15, "right_tool")]:
        js = agibot_gdk.JointStates()
        js.group, js.target_type, js.nums = group, "omnipicker", 1
        s = agibot_gdk.JointState()
        s.position = clip_gripper(target[idx])
        js.states = [s]
        robot.move_ee_pos(js)


def main():
    print("🤖 初始化 GDK...")
    agibot_gdk.gdk_init()
    robot = agibot_gdk.Robot()
    time.sleep(2)

    current = get_qpos(robot)
    target = HOME_QPOS.copy()

    print(f"当前位置: arm=[{current[:14].min():.3f},{current[:14].max():.3f}] grip=[{current[14]:.3f},{current[15]:.3f}]")
    print(f"目标位置: arm=[{target[:14].min():.3f},{target[:14].max():.3f}] grip=[{target[14]:.3f},{target[15]:.3f}]")
    max_delta = np.abs(target - current).max()
    est_time = max_delta / ARM_VELOCITY
    print(f"最大关节变化: {max_delta:.3f} rad ({np.degrees(max_delta):.1f}°), 预计~{est_time:.1f}s")

    input(f"\n✋ 按 Enter 开始归位 (先抬臂再归位, vel={ARM_VELOCITY} rad/s)...\n")

    try:
        # 1) 先抬臂: 只动 joint2/joint4 让末端远离桌面
        lift_positions = [float(x) for x in LIFT_QPOS]
        lift_velocities = [LIFT_VELOCITY] * 14
        print(f"🔄 阶段1: 抬臂远离桌面 (move_arm_joint)...")
        t0 = time.time()
        result = robot.move_arm_joint(lift_positions, lift_velocities, 2)
        dt = time.time() - t0
        print(f"  ✅ 抬臂完成: {result}, 耗时 {dt:.1f}s")

        # 2) 再归位: 从抬臂位移到 HOME
        arm_positions = [float(x) for x in target[:14]]
        arm_velocities = [ARM_VELOCITY] * 14
        print(f"🔄 阶段2: 归位到 HOME (move_arm_joint)...")
        t0 = time.time()
        result = robot.move_arm_joint(arm_positions, arm_velocities, 2)  # 2=双臂
        dt = time.time() - t0
        print(f"  ✅ 归位完成: {result}, 耗时 {dt:.1f}s")

        # 2) 夹爪归位 (持续发送3秒)
        print("🤏 夹爪归位 (持续3s)...")
        for _ in range(300):
            send_gripper(robot, target)
            time.sleep(0.01)

    except Exception as e:
        print(f"❌ 归位异常: {e}")
        import traceback; traceback.print_exc()

    # 验证最终位置
    final = get_qpos(robot)
    error = np.abs(final - target).max()
    print(f"✅ 归位完成! 最大误差: {error:.4f} rad ({np.degrees(error):.2f}°)")
    print(f"  最终位置: {[f'{x:.3f}' for x in final]}")


if __name__ == '__main__':
    main()
