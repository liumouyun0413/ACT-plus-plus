#!/usr/bin/env python3
"""Audit ROS 2 MCAP episodes for stream gaps and ACT++ conversion readiness.

Usage:
    python check_mcap_dataset_consistency.py [dataset_dir]

Requires: pip install mcap mcap-ros2-support pyyaml
"""

from __future__ import annotations

import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import yaml
from mcap_ros2.reader import read_ros2_messages


ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "fold_clothes"
REPORT = ROOT / "mcap_consistency_report.txt"

CAMERAS = {
    "/camera/head/color/image_raw/compressed": "head_color",
    "/camera/left_wrist/color/image_raw/compressed": "hand_left_color",
    "/camera/right_wrist/color/image_raw/compressed": "hand_right_color",
}
REQUIRED = set(CAMERAS) | {
    "/hal/joint_states",
    "/lead/joint_states",
    "/motion_ctl/gripper/left/state",
    "/motion_ctl/gripper/right/state",
    "/motion_ctl/gripper/left",
    "/motion_ctl/gripper/right",
}
EXPECTED_DIMS = {
    "/hal/joint_states": 34,
    "/lead/joint_states": 14,
    "/motion_ctl/gripper/left/state": 1,
    "/motion_ctl/gripper/right/state": 1,
}


def percentile(values, q):
    if not values:
        return float("nan")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def stamp_seconds(msg, fallback_ns):
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return fallback_ns / 1e9
    return float(stamp.sec) + float(stamp.nanosec) / 1e9


def finite_sequence(value):
    try:
        return all(math.isfinite(float(x)) for x in value)
    except (TypeError, ValueError):
        return False


def inspect_episode(ep_dir: Path):
    metadata_path = ep_dir / "metadata.yaml"
    errors, warnings = [], []
    if not metadata_path.exists():
        return None, ["缺少 metadata.yaml"], []
    metadata = yaml.safe_load(metadata_path.read_text())
    info = metadata.get("rosbag2_bagfile_information", {})
    relative = info.get("relative_file_paths", [])
    if len(relative) != 1:
        errors.append(f"metadata 中 MCAP 文件数应为 1，实际 {len(relative)}")
        return None, errors, warnings
    mcap_path = ep_dir / relative[0]
    if not mcap_path.exists() or mcap_path.stat().st_size == 0:
        errors.append(f"MCAP 不存在或为空: {mcap_path.name}")
        return None, errors, warnings

    times = defaultdict(list)
    bag_times = defaultdict(list)
    bad_finite = defaultdict(int)
    bad_dims = defaultdict(int)
    bad_jpeg = defaultdict(int)
    names = defaultdict(set)
    image_sizes = defaultdict(set)
    total = 0

    try:
        for record in read_ros2_messages(str(mcap_path)):
            topic, msg = record.channel.topic, record.ros_msg
            total += 1
            bag_t = record.log_time_ns / 1e9
            bag_times[topic].append(bag_t)
            times[topic].append(stamp_seconds(msg, record.log_time_ns))

            if hasattr(msg, "name"):
                names[topic].add(tuple(msg.name))
            if hasattr(msg, "position"):
                pos = msg.position
                if isinstance(pos, (list, tuple)):
                    if not finite_sequence(pos):
                        bad_finite[topic] += 1
                    expected = EXPECTED_DIMS.get(topic)
                    if expected is not None and len(pos) != expected:
                        bad_dims[topic] += 1
                elif not math.isfinite(float(pos)):
                    bad_finite[topic] += 1
            if hasattr(msg, "velocity") and msg.velocity and not finite_sequence(msg.velocity):
                bad_finite[topic] += 1
            if topic in CAMERAS:
                data = bytes(msg.data)
                image_sizes[topic].add(len(data))
                if len(data) < 4 or not data.startswith(b"\xff\xd8") or not data.endswith(b"\xff\xd9"):
                    bad_jpeg[topic] += 1
            if topic.endswith("/gripper/left") or topic.endswith("/gripper/right"):
                if not math.isfinite(float(msg.position)) or not math.isfinite(float(msg.max_effort)):
                    bad_finite[topic] += 1
    except Exception as exc:
        errors.append(f"MCAP 解码失败: {type(exc).__name__}: {exc}")

    missing = sorted(REQUIRED - set(times))
    if missing:
        errors.append("缺少必需主题: " + ", ".join(missing))
    metadata_count = int(info.get("message_count", -1))
    if metadata_count != total:
        errors.append(f"消息总数不符: metadata={metadata_count}, decoded={total}")

    stats = {}
    for topic, ts in sorted(times.items()):
        backwards = sum(b < a for a, b in zip(ts, ts[1:]))
        dts = [b - a for a, b in zip(ts, ts[1:]) if b >= a]
        med = statistics.median(dts) if dts else float("nan")
        # A gap over 3 nominal periods is a likely interruption; 0.2 s is always material.
        gap_threshold = max(3.0 * med, 0.2) if dts else float("nan")
        gaps = [(i, dt) for i, dt in enumerate(dts) if dt > gap_threshold]
        duration = ts[-1] - ts[0] if len(ts) > 1 else 0.0
        rate = (len(ts) - 1) / duration if duration > 0 else 0.0
        stats[topic] = {
            "count": len(ts), "rate": rate, "median": med,
            "p99": percentile(dts, 0.99), "max": max(dts, default=float("nan")),
            "gaps": gaps, "backwards": backwards,
        }
        if backwards:
            errors.append(f"{topic}: header 时间戳倒退 {backwards} 次")
        if gaps:
            warnings.append(f"{topic}: 疑似断流 {len(gaps)} 次，最大间隔 {max(x[1] for x in gaps):.3f}s")
        if bad_finite[topic]:
            errors.append(f"{topic}: NaN/Inf 消息 {bad_finite[topic]} 条")
        if bad_dims[topic]:
            errors.append(f"{topic}: position 维度异常 {bad_dims[topic]} 条")
        if bad_jpeg[topic]:
            errors.append(f"{topic}: JPEG 首尾标记异常 {bad_jpeg[topic]} 条")
        if len(names[topic]) > 1:
            errors.append(f"{topic}: JointState name/schema 发生变化 ({len(names[topic])} 种)")

    if all(t in stats for t in CAMERAS):
        rates = {t: stats[t]["rate"] for t in CAMERAS}
        fastest = max(rates.values())
        for topic, rate in rates.items():
            if rate < fastest * 0.8:
                warnings.append(
                    f"{topic}: 平均 {rate:.2f}Hz，明显低于最快相机 {fastest:.2f}Hz；转换时需重采样/重复帧"
                )

    # Common interval is the only portion that can be aligned into ACT timesteps.
    required_present = [times[t] for t in REQUIRED if t in times and times[t]]
    overlap = None
    if required_present:
        start = max(x[0] for x in required_present)
        end = min(x[-1] for x in required_present)
        overlap = max(0.0, end - start)
        bag_duration = float(info.get("duration", {}).get("nanoseconds", 0)) / 1e9
        if overlap < bag_duration * 0.95:
            warnings.append(f"必需主题共同有效区间仅 {overlap:.3f}s / bag {bag_duration:.3f}s")

    return {
        "file": mcap_path.name,
        "size": mcap_path.stat().st_size,
        "bag_duration": float(info.get("duration", {}).get("nanoseconds", 0)) / 1e9,
        "decoded": total,
        "stats": stats,
        "overlap": overlap,
        "bad_jpeg": dict(bad_jpeg),
        "image_sizes": {k: (min(v), max(v)) for k, v in image_sizes.items() if v},
    }, errors, warnings


def main():
    episodes = sorted(p for p in ROOT.glob("episode_*") if p.is_dir())
    lines = [f"MCAP 数据集审核: {ROOT.resolve()}", f"episode 数: {len(episodes)}", ""]
    all_errors, all_warnings = 0, 0
    for ep in episodes:
        result, errors, warnings = inspect_episode(ep)
        all_errors += len(errors)
        all_warnings += len(warnings)
        lines += ["=" * 90, ep.name]
        if result:
            overlap_text = (
                f"{result['overlap']:.3f}s"
                if result['overlap'] is not None else "不可用"
            )
            lines.append(
                f"文件={result['file']}  大小={result['size']/1024**2:.1f}MiB  "
                f"时长={result['bag_duration']:.3f}s  解码消息={result['decoded']}  "
                f"共同有效区间={overlap_text}"
            )
            lines.append("主题统计（header 时间戳）:")
            for topic, s in result["stats"].items():
                lines.append(
                    f"  {topic}: n={s['count']}, rate={s['rate']:.2f}Hz, "
                    f"dt_med={s['median']*1000:.2f}ms, p99={s['p99']*1000:.2f}ms, "
                    f"max={s['max']*1000:.2f}ms, gaps={len(s['gaps'])}"
                )
        lines.append("错误:")
        lines += [f"  [ERROR] {x}" for x in errors] or ["  无"]
        lines.append("警告:")
        lines += [f"  [WARN] {x}" for x in warnings] or ["  无"]
        lines.append("")

    lines += ["=" * 90, "ACT++ 兼容性结论"]
    lines.append("原始 MCAP 不能被当前 utils.py 直接训练，必须先转换为 ACT HDF5。")
    lines.append("建议映射: hal 前14关节+左右夹爪状态 -> 16维 qpos；lead 14关节+左右夹爪命令 -> 16维 action。")
    lines.append("qvel 使用 hal 前14维速度，夹爪速度可差分或置零；角度需统一转换为弧度。")
    lines.append("相机映射: head/left_wrist/right_wrist -> head_color/hand_left_color/hand_right_color。")
    lines.append("转换时以共同有效区间建立固定时间轴，并对各流做最近邻/保持采样；不得直接按消息序号对齐。")
    lines.append("")
    lines.append(f"汇总: ERROR={all_errors}, WARN={all_warnings}")
    lines.append("判定: " + ("存在硬错误，不应进入训练" if all_errors else "无硬错误；完成时间同步转换并处理警告后可用于训练"))
    REPORT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n报告已保存: {REPORT}")


if __name__ == "__main__":
    main()
