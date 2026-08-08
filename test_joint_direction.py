#!/usr/bin/env python3
"""测试 joint2/joint4 方向: 小幅变化观察末端是否向上"""
import time
import numpy as np
import agibot_gdk

ARM_JOINT_INDICES = list(range(8, 22))
ARM_VELOCITY = 0.1  # 慢速


def get_arm_positions(robot):
    js = robot.get_joint_states()
    positions = [s['motor_position'] for s in js['states']]
    return [positions[i] for i in ARM_JOINT_INDICES]


def main():
    print("🤖 初始化 GDK...")
    agibot_gdk.gdk_init()
    robot = agibot_gdk.Robot()
    time.sleep(2)

    current = get_arm_positions(robot)
    print(f"当前14关节: {[f'{x:.3f}' for x in current]}")
    print(f"  左臂: j1={current[0]:.3f} j2={current[1]:.3f} j3={current[2]:.3f} j4={current[3]:.3f}")
    print(f"  右臂: j1={current[7]:.3f} j2={current[8]:.3f} j3={current[9]:.3f} j4={current[10]:.3f}")

    # 测试1: joint1 增加 0.2 rad
    print("\n===== 测试1: 左j1 +0.2, 右j1 -0.2 (对称, 观察是否向上) =====")
    input("按 Enter 执行...")
    target1 = list(current)
    target1[0] += 0.2   # 左 joint1
    target1[7] -= 0.2   # 右 joint1 (左右对称, 方向相反)
    print(f"  左j1: {current[0]:.3f} → {target1[0]:.3f}")
    print(f"  右j1: {current[7]:.3f} → {target1[7]:.3f}")
    robot.move_arm_joint(target1, [ARM_VELOCITY]*14, 2)
    print("  ✅ 完成, 观察末端是否向上移动了")

    # 回到原位
    input("\n按 Enter 回到原位...")
    robot.move_arm_joint(list(current), [ARM_VELOCITY]*14, 2)
    print("  ✅ 已回原位")

    # 测试2: joint1 减少 0.2 rad
    print("\n===== 测试2: 左j1 -0.2, 右j1 +0.2 (对称, 观察方向) =====")
    input("按 Enter 执行...")
    target2 = list(current)
    target2[0] -= 0.2
    target2[7] += 0.2
    print(f"  左j1: {current[0]:.3f} → {target2[0]:.3f}")
    print(f"  右j1: {current[7]:.3f} → {target2[7]:.3f}")
    robot.move_arm_joint(target2, [ARM_VELOCITY]*14, 2)
    print("  ✅ 完成, 观察方向")

    # 回到原位
    input("\n按 Enter 回到原位...")
    robot.move_arm_joint(list(current), [ARM_VELOCITY]*14, 2)
    print("  ✅ 已回原位")

    # 测试3: joint1 不对称, 同方向 +0.2
    print("\n===== 测试3: 左j1 +0.2, 右j1 +0.2 (同方向, 看是否一起向上) =====")
    input("按 Enter 执行...")
    target3 = list(current)
    target3[0] += 0.2
    target3[7] += 0.2
    print(f"  左j1: {current[0]:.3f} → {target3[0]:.3f}")
    print(f"  右j1: {current[7]:.3f} → {target3[7]:.3f}")
    robot.move_arm_joint(target3, [ARM_VELOCITY]*14, 2)
    print("  ✅ 完成, 观察方向")

    input("\n按 Enter 回到原位并结束...")
    robot.move_arm_joint(list(current), [ARM_VELOCITY]*14, 2)
    print("✅ 测试结束, 已回原位")


if __name__ == '__main__':
    main()
