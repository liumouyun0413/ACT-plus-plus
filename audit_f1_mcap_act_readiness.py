#!/usr/bin/env python3
"""Audit F1 ROS 2 MCAP recordings for dropped frames and ACT++ readiness.

Continuity is measured with MCAP log_time. Message header stamps are audited
separately because camera drivers can emit stale stamps at recording startup.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml
from mcap_ros2.reader import read_ros2_messages

CAMERAS = (
    "/camera/head/color/image_raw/compressed",
    "/camera/left_wrist/color/image_raw/compressed",
    "/camera/right_wrist/color/image_raw/compressed",
)
STATE_TOPICS = (
    "/hal/joint_states",
    "/motion_ctl/gripper/left/state",
    "/motion_ctl/gripper/right/state",
)
ACTION_TOPICS = (
    "/lead/joint_states",
    "/motion_ctl/gripper/left",
    "/motion_ctl/gripper/right",
)
REQUIRED = CAMERAS + STATE_TOPICS + ACTION_TOPICS
EXPECTED_POSITION_DIMS = {
    "/hal/joint_states": 34,
    "/lead/joint_states": 14,
}


def header_time(msg: Any) -> float | None:
    stamp = getattr(getattr(msg, "header", None), "stamp", None)
    if stamp is None:
        return None
    return float(stamp.sec) + float(stamp.nanosec) / 1e9


def all_finite(values: Any) -> bool:
    try:
        return all(math.isfinite(float(value)) for value in values)
    except (TypeError, ValueError):
        return False


def nearest_distance(ordered: list[float], target: float) -> float:
    index = bisect_left(ordered, target)
    distances = []
    if index < len(ordered):
        distances.append(abs(ordered[index] - target))
    if index:
        distances.append(abs(ordered[index - 1] - target))
    return min(distances, default=math.inf)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def topic_metrics(log_times: list[float], header_times: list[float]) -> dict[str, Any]:
    positive_dt = [b - a for a, b in zip(log_times, log_times[1:]) if b >= a]
    median_dt = statistics.median(positive_dt) if positive_dt else None
    drop_threshold = 1.5 * median_dt if median_dt else math.inf
    drop_dts = [dt for dt in positive_dt if dt > drop_threshold]
    estimated_missing = (
        sum(max(1, round(dt / median_dt) - 1) for dt in drop_dts)
        if median_dt else 0
    )
    severe_dts = [dt for dt in positive_dt if dt > 0.2]
    header_backwards = sum(b < a for a, b in zip(header_times, header_times[1:]))
    header_jump_count = 0
    max_header_log_offset_change = 0.0
    if len(header_times) == len(log_times) and len(log_times) > 1:
        offsets = [header - log for header, log in zip(header_times, log_times)]
        offset_changes = [abs(b - a) for a, b in zip(offsets, offsets[1:])]
        header_jump_count = sum(change > 0.2 for change in offset_changes)
        max_header_log_offset_change = max(offset_changes, default=0.0)
    duration = log_times[-1] - log_times[0] if len(log_times) > 1 else 0.0
    return {
        "count": len(log_times),
        "rate_hz": (len(log_times) - 1) / duration if duration > 0 else 0.0,
        "median_dt_ms": median_dt * 1000 if median_dt is not None else None,
        "p99_dt_ms": (percentile(positive_dt, 0.99) or 0.0) * 1000,
        "max_dt_ms": max(positive_dt, default=0.0) * 1000,
        "drop_events": len(drop_dts),
        "estimated_missing": estimated_missing,
        "severe_gaps": len(severe_dts),
        "max_severe_gap_s": max(severe_dts, default=0.0),
        "log_time_backwards": sum(b < a for a, b in zip(log_times, log_times[1:])),
        "header_time_backwards": header_backwards,
        "header_clock_jumps": header_jump_count,
        "max_header_log_offset_change_s": max_header_log_offset_change,
    }


def inspect_episode(dataset: str, episode_dir: Path, alignment_tolerance_s: float) -> dict[str, Any]:
    result: dict[str, Any] = {
        "dataset": dataset,
        "episode": episode_dir.name,
        "path": str(episode_dir),
        "errors": [],
        "warnings": [],
        "topics": {},
    }
    metadata_path = episode_dir / "metadata.yaml"
    if not metadata_path.exists():
        result["errors"].append("missing metadata.yaml")
        result["classification"] = "REJECT"
        return result

    try:
        info = yaml.safe_load(metadata_path.read_text())["rosbag2_bagfile_information"]
        relative_paths = info.get("relative_file_paths", [])
        if len(relative_paths) != 1:
            raise ValueError(f"expected one MCAP, got {len(relative_paths)}")
        mcap_path = episode_dir / relative_paths[0]
        if not mcap_path.is_file() or mcap_path.stat().st_size == 0:
            raise ValueError(f"missing/empty MCAP: {mcap_path.name}")
    except Exception as exc:
        result["errors"].append(f"metadata/file error: {exc}")
        result["classification"] = "REJECT"
        return result

    log_times: dict[str, list[float]] = defaultdict(list)
    header_times: dict[str, list[float]] = defaultdict(list)
    invalid_numeric: dict[str, int] = defaultdict(int)
    invalid_dims: dict[str, int] = defaultdict(int)
    invalid_jpeg: dict[str, int] = defaultdict(int)
    joint_names: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    decoded = 0

    try:
        for record in read_ros2_messages(str(mcap_path)):
            decoded += 1
            topic = record.channel.topic
            if topic not in REQUIRED:
                continue
            msg = record.ros_msg
            log_times[topic].append(record.log_time_ns / 1e9)
            stamp = header_time(msg)
            if stamp is not None:
                header_times[topic].append(stamp)
            if topic in CAMERAS:
                data = bytes(msg.data)
                if len(data) < 4 or not data.startswith(b"\xff\xd8") or not data.endswith(b"\xff\xd9"):
                    invalid_jpeg[topic] += 1
            if topic in EXPECTED_POSITION_DIMS:
                position = msg.position
                if len(position) != EXPECTED_POSITION_DIMS[topic]:
                    invalid_dims[topic] += 1
                if not all_finite(position):
                    invalid_numeric[topic] += 1
                joint_names[topic].add(tuple(msg.name))
            elif topic in STATE_TOPICS:
                position = getattr(msg, "position", ())
                if isinstance(position, (list, tuple)):
                    if len(position) != 1 or not all_finite(position):
                        invalid_numeric[topic] += 1
                elif not math.isfinite(float(position)):
                    invalid_numeric[topic] += 1
            elif topic in ACTION_TOPICS[1:]:
                if not math.isfinite(float(msg.position)) or not math.isfinite(float(msg.max_effort)):
                    invalid_numeric[topic] += 1
    except Exception as exc:
        result["errors"].append(f"MCAP decode failed: {type(exc).__name__}: {exc}")

    result.update({
        "mcap": mcap_path.name,
        "size_bytes": mcap_path.stat().st_size,
        "bag_duration_s": float(info.get("duration", {}).get("nanoseconds", 0)) / 1e9,
        "metadata_messages": int(info.get("message_count", -1)),
        "decoded_messages": decoded,
    })
    if result["metadata_messages"] != decoded:
        result["errors"].append(
            f"message count mismatch metadata={result['metadata_messages']} decoded={decoded}"
        )

    missing_topics = [topic for topic in REQUIRED if not log_times[topic]]
    if missing_topics:
        result["errors"].append("missing topics: " + ", ".join(missing_topics))

    total_drop_events = total_estimated_missing = total_severe_gaps = 0
    header_clock_topics = 0
    for topic in REQUIRED:
        if not log_times[topic]:
            continue
        metrics = topic_metrics(log_times[topic], header_times[topic])
        metrics.update({
            "invalid_numeric": invalid_numeric[topic],
            "invalid_dims": invalid_dims[topic],
            "invalid_jpeg": invalid_jpeg[topic],
            "joint_name_schemas": len(joint_names[topic]),
        })
        result["topics"][topic] = metrics
        total_drop_events += metrics["drop_events"]
        total_estimated_missing += metrics["estimated_missing"]
        total_severe_gaps += metrics["severe_gaps"]
        if metrics["header_clock_jumps"] or metrics["header_time_backwards"]:
            header_clock_topics += 1
        if metrics["log_time_backwards"]:
            result["errors"].append(f"{topic}: log_time backwards")
        if metrics["invalid_numeric"] or metrics["invalid_dims"] or metrics["invalid_jpeg"]:
            result["errors"].append(
                f"{topic}: numeric={metrics['invalid_numeric']} dims={metrics['invalid_dims']} "
                f"jpeg={metrics['invalid_jpeg']}"
            )
        if metrics["joint_name_schemas"] > 1:
            result["errors"].append(f"{topic}: changing JointState schema")

    result["drop_events"] = total_drop_events
    result["estimated_missing"] = total_estimated_missing
    result["severe_gaps"] = total_severe_gaps
    result["header_clock_topics"] = header_clock_topics

    overlap_s = 0.0
    alignment_coverage = 0.0
    max_nearest_ms = None
    if not missing_topics:
        overlap_start = max(log_times[topic][0] for topic in REQUIRED)
        overlap_end = min(log_times[topic][-1] for topic in REQUIRED)
        overlap_s = max(0.0, overlap_end - overlap_start)
        targets = [
            value for value in log_times["/hal/joint_states"]
            if overlap_start <= value <= overlap_end
        ]
        nearest = [
            max(nearest_distance(log_times[topic], target) for topic in REQUIRED)
            for target in targets
        ]
        alignment_coverage = (
            sum(distance <= alignment_tolerance_s for distance in nearest) / len(nearest)
            if nearest else 0.0
        )
        max_nearest_ms = max(nearest, default=0.0) * 1000
    result["common_overlap_s"] = overlap_s
    result["alignment_coverage_50ms"] = alignment_coverage
    result["max_nearest_ms"] = max_nearest_ms

    hard_failure = bool(result["errors"]) or overlap_s < 5.0 or alignment_coverage < 0.95
    if hard_failure:
        result["classification"] = "REJECT"
    elif total_severe_gaps:
        result["classification"] = "FRAGMENT_ONLY"
        result["warnings"].append("one or more required streams has a gap >200 ms")
    else:
        result["classification"] = "READY"
    if total_drop_events:
        result["warnings"].append(
            f"{total_drop_events} dropped-frame events, about {total_estimated_missing} missing messages"
        )
    if header_clock_topics:
        result["warnings"].append(
            f"header clock anomaly on {header_clock_topics} topics; convert using MCAP log_time"
        )
    return result


def write_reports(results: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "mcap_act_readiness.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n"
    )
    columns = [
        "dataset", "episode", "classification", "bag_duration_s", "common_overlap_s",
        "alignment_coverage_50ms", "drop_events", "estimated_missing", "severe_gaps",
        "header_clock_topics", "size_bytes", "errors", "warnings", "path",
    ]
    with (output_dir / "mcap_act_readiness.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for item in results:
            row = {key: item.get(key) for key in columns}
            row["errors"] = " | ".join(item["errors"])
            row["warnings"] = " | ".join(item["warnings"])
            writer.writerow(row)

    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    aggregate = defaultdict(int)
    for item in results:
        counts[item["dataset"]][item["classification"]] += 1
        aggregate[item["classification"]] += 1
    lines = [
        "# F1 MCAP 丢帧与 ACT++ HDF5 转换条件审核", "",
        f"- 实际审核 episode：**{len(results)}**",
        "- 真实丢帧依据：MCAP `log_time` 相邻间隔 > 1.5×该主题中位周期",
        "- 严重断流：任一必需主题相邻间隔 > 200 ms",
        "- 对齐可行性：以 `/hal/joint_states` 为时间轴，全部必需流在 ±50 ms 内可找到样本",
        "- `READY`：整条可转换；`FRAGMENT_ONLY`：应按严重断流切片后转换；`REJECT`：存在硬错误或对齐覆盖不足",
        "",
        "## 分类汇总", "",
        "| 数据集 | READY | FRAGMENT_ONLY | REJECT | 合计 |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset in sorted(counts):
        c = counts[dataset]
        total = sum(c.values())
        lines.append(
            f"| {dataset} | {c['READY']} | {c['FRAGMENT_ONLY']} | {c['REJECT']} | {total} |"
        )
    lines.append(
        f"| **总计** | **{aggregate['READY']}** | **{aggregate['FRAGMENT_ONLY']}** | "
        f"**{aggregate['REJECT']}** | **{len(results)}** |"
    )
    total_events = sum(item.get("drop_events", 0) for item in results)
    total_missing = sum(item.get("estimated_missing", 0) for item in results)
    severe_episodes = sum(item.get("severe_gaps", 0) > 0 for item in results)
    header_episodes = sum(item.get("header_clock_topics", 0) > 0 for item in results)
    lines += [
        "", "## 丢帧汇总", "",
        f"- 丢帧事件：**{total_events}**",
        f"- 估算缺失消息：**{total_missing}**（跨所有9个必需流，不能等同于缺失ACT时间步）",
        f"- 含 >200 ms 严重断流的episode：**{severe_episodes}**",
        f"- 含header时钟异常的episode：**{header_episodes}**",
        "", "## 转换要求", "",
        "1. 必须按MCAP `log_time`同步，不能直接按消息序号或异常的相机header时间对齐。",
        "2. 建议以 `/hal/joint_states` 约30 Hz建立时间轴，三相机和action/state流做最近邻或保持采样。",
        "3. `qpos`：HAL前14关节（度转弧度）+左右夹爪状态；`action`：lead 14关节（度转弧度）+左右夹爪命令。",
        "4. 三相机映射为 `head_color`、`hand_left_color`、`hand_right_color`，解码后BGR转RGB并统一尺寸。",
        "5. `FRAGMENT_ONLY`不能跨断流拼成一条轨迹，应切成独立连续HDF5；`REJECT`需查看CSV的errors字段。",
        "", "## 需重点检查的episode", "",
    ]
    notable = sorted(
        (item for item in results if item["classification"] != "READY"),
        key=lambda item: (item["classification"], -item.get("severe_gaps", 0), item["dataset"], item["episode"]),
    )
    if notable:
        for item in notable:
            detail = "; ".join(item["errors"] + item["warnings"])
            lines.append(f"- `{item['dataset']}/{item['episode']}` — **{item['classification']}**：{detail}")
    else:
        lines.append("- 无。")
    (output_dir / "mcap_act_readiness_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alignment-tolerance-ms", type=float, default=50.0)
    args = parser.parse_args()

    episode_dirs: list[tuple[str, Path]] = []
    for dataset_dir in args.datasets:
        episodes = sorted(path for path in dataset_dir.glob("episode_*") if path.is_dir())
        episode_dirs.extend((dataset_dir.name, path) for path in episodes)
        print(f"{dataset_dir}: {len(episodes)} episodes", flush=True)

    results = []
    for index, (dataset, episode_dir) in enumerate(episode_dirs, 1):
        item = inspect_episode(dataset, episode_dir, args.alignment_tolerance_ms / 1000.0)
        results.append(item)
        print(
            f"[{index:03}/{len(episode_dirs)}] {dataset}/{episode_dir.name}: "
            f"{item['classification']} drops={item.get('drop_events', 0)} "
            f"severe={item.get('severe_gaps', 0)}",
            flush=True,
        )
    write_reports(results, args.output_dir)
    print(f"Reports: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
