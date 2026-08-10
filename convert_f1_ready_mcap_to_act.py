#!/usr/bin/env python3
"""Convert audited READY F1 MCAP episodes to ACT++ HDF5.

Synchronization uses MCAP log_time only.  /hal/joint_states is the ~30 Hz
reference clock. Short gaps are handled by nearest-neighbour/zero-order hold;
READY inputs are guaranteed to have no required-stream gap over 200 ms.
Arm lag is estimated from motion derivatives per source. Only sources whose lag
is an outlier relative to the dataset median are corrected; gripper columns are
never shifted.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import yaml
from mcap_ros2.reader import read_ros2_messages

CAMERA_TOPICS = {
    "/camera/left_wrist/color/image_raw/compressed": "hand_left_color",
    "/camera/right_wrist/color/image_raw/compressed": "hand_right_color",
    "/camera/head/color/image_raw/compressed": "head_color",
}
HAL = "/hal/joint_states"
LEAD = "/lead/joint_states"
LEFT_STATE = "/motion_ctl/gripper/left/state"
RIGHT_STATE = "/motion_ctl/gripper/right/state"
LEFT_COMMAND = "/motion_ctl/gripper/left"
RIGHT_COMMAND = "/motion_ctl/gripper/right"
REQUIRED = tuple(CAMERA_TOPICS) + (
    HAL, LEAD, LEFT_STATE, RIGHT_STATE, LEFT_COMMAND, RIGHT_COMMAND,
)
ARM_NAMES = tuple(
    [f"arm_l_j{i}" for i in range(1, 8)] +
    [f"arm_r_j{i}" for i in range(1, 8)]
)


def nearest_index(times: np.ndarray, target: float) -> tuple[int, float]:
    index = int(np.searchsorted(times, target))
    choices = []
    if index < len(times):
        choices.append(index)
    if index:
        choices.append(index - 1)
    best = min(choices, key=lambda i: abs(float(times[i]) - target))
    return best, abs(float(times[best]) - target)


def previous_index(times: np.ndarray, target: float) -> tuple[int, float]:
    index = max(0, int(np.searchsorted(times, target, side="right")) - 1)
    return index, abs(float(times[index]) - target)


def arm_indices(names: list[str]) -> list[int]:
    mapping = {name: index for index, name in enumerate(names)}
    if all(name in mapping for name in ARM_NAMES):
        return [mapping[name] for name in ARM_NAMES]
    if len(names) >= 14:
        return list(range(14))
    raise ValueError(f"cannot map 14 arm joints from names={names}")


def read_episode(episode_dir: Path) -> dict[str, Any]:
    metadata = yaml.safe_load((episode_dir / "metadata.yaml").read_text())
    info = metadata["rosbag2_bagfile_information"]
    mcap_path = episode_dir / info["relative_file_paths"][0]
    streams: dict[str, dict[str, list[Any]]] = {
        topic: {"times": [], "values": []} for topic in REQUIRED
    }
    arm_velocity: list[np.ndarray] = []

    for record in read_ros2_messages(str(mcap_path)):
        topic = record.channel.topic
        if topic not in streams:
            continue
        msg = record.ros_msg
        streams[topic]["times"].append(record.log_time_ns / 1e9)
        if topic in CAMERA_TOPICS:
            streams[topic]["values"].append(bytes(msg.data))
        elif topic in (HAL, LEAD):
            indices = arm_indices(list(msg.name))
            streams[topic]["values"].append(
                np.asarray(msg.position, dtype=np.float64)[indices]
            )
            if topic == HAL:
                velocity = np.asarray(msg.velocity, dtype=np.float64)
                if len(velocity) > max(indices):
                    arm_velocity.append(velocity[indices])
                else:
                    arm_velocity.append(np.full(14, np.nan))
        elif topic in (LEFT_STATE, RIGHT_STATE):
            position = msg.position
            value = position[0] if isinstance(position, (list, tuple)) else position
            streams[topic]["values"].append(float(value))
        else:
            streams[topic]["values"].append(float(msg.position))

    for topic in REQUIRED:
        if not streams[topic]["times"]:
            raise ValueError(f"missing required topic: {topic}")
        streams[topic]["times"] = np.asarray(streams[topic]["times"], dtype=np.float64)
        if topic not in CAMERA_TOPICS:
            streams[topic]["values"] = np.asarray(streams[topic]["values"], dtype=np.float64)
    streams[HAL]["velocity"] = np.asarray(arm_velocity, dtype=np.float64)
    return {"mcap_path": mcap_path, "streams": streams}


def common_reference_times(streams: dict[str, dict[str, Any]]) -> np.ndarray:
    start = max(float(streams[topic]["times"][0]) for topic in REQUIRED)
    end = min(float(streams[topic]["times"][-1]) for topic in REQUIRED)
    reference = streams[HAL]["times"]
    return reference[(reference >= start) & (reference <= end)]


def align_numeric(
    stream: dict[str, Any], reference: np.ndarray, hold: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    values = []
    distances = []
    times = stream["times"]
    for target in reference:
        index, distance = previous_index(times, target) if hold else nearest_index(times, target)
        values.append(stream["values"][index])
        distances.append(distance)
    return np.asarray(values), np.asarray(distances)


def align_episode(raw: dict[str, Any]) -> dict[str, Any]:
    streams = raw["streams"]
    reference = common_reference_times(streams)
    if len(reference) < 150:
        raise ValueError(f"only {len(reference)} common frames")

    qpos_arm, hal_distance = align_numeric(streams[HAL], reference)
    lead_arm_deg, lead_distance = align_numeric(streams[LEAD], reference, hold=True)
    left_state, left_state_distance = align_numeric(streams[LEFT_STATE], reference)
    right_state, right_state_distance = align_numeric(streams[RIGHT_STATE], reference)
    left_command, left_command_distance = align_numeric(streams[LEFT_COMMAND], reference, hold=True)
    right_command, right_command_distance = align_numeric(streams[RIGHT_COMMAND], reference, hold=True)

    hal_indices = np.searchsorted(streams[HAL]["times"], reference)
    hal_indices = np.clip(hal_indices, 0, len(streams[HAL]["times"]) - 1)
    qvel_arm = streams[HAL]["velocity"][hal_indices]
    if not np.all(np.isfinite(qvel_arm)):
        dt = np.gradient(reference)
        qvel_arm = np.gradient(np.deg2rad(qpos_arm), axis=0) / dt[:, None]
    else:
        qvel_arm = np.deg2rad(qvel_arm)

    images: dict[str, list[bytes]] = {}
    all_distances = [
        hal_distance, lead_distance, left_state_distance, right_state_distance,
        left_command_distance, right_command_distance,
    ]
    for topic, camera_name in CAMERA_TOPICS.items():
        camera_values = []
        camera_distances = []
        for target in reference:
            index, distance = nearest_index(streams[topic]["times"], target)
            camera_values.append(streams[topic]["values"][index])
            camera_distances.append(distance)
        images[camera_name] = camera_values
        all_distances.append(np.asarray(camera_distances))

    qpos = np.concatenate([
        np.deg2rad(qpos_arm), left_state[:, None], right_state[:, None]
    ], axis=1).astype(np.float32)
    qvel = np.concatenate([
        qvel_arm, np.zeros((len(reference), 2), dtype=np.float64)
    ], axis=1).astype(np.float32)
    action = np.concatenate([
        np.deg2rad(lead_arm_deg), left_command[:, None], right_command[:, None]
    ], axis=1).astype(np.float32)
    max_distance = np.max(np.stack(all_distances), axis=0)
    return {
        "times": reference,
        "qpos": qpos,
        "qvel": qvel,
        "action": action,
        "images": images,
        "max_sync_distance": max_distance,
    }


def smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    kernel = np.ones(window, dtype=np.float64) / window
    return np.stack([
        np.convolve(values[:, column], kernel, mode="same")
        for column in range(values.shape[1])
    ], axis=1)


def estimate_arm_lag(qpos: np.ndarray, action: np.ndarray, max_lag: int = 15) -> tuple[int, list[float]]:
    q_velocity = np.diff(smooth(qpos[:, :14]), axis=0)
    a_velocity = np.diff(smooth(action[:, :14]), axis=0)
    scores = []
    for lag in range(max_lag + 1):
        q = q_velocity[lag:] if lag else q_velocity
        a = a_velocity[:-lag] if lag else a_velocity
        joint_scores = []
        for joint in range(14):
            qj, aj = q[:, joint], a[:, joint]
            moving = (np.abs(qj) > np.percentile(np.abs(qj), 35)) | (
                np.abs(aj) > np.percentile(np.abs(aj), 35)
            )
            if moving.sum() < 30 or np.std(qj[moving]) < 1e-5 or np.std(aj[moving]) < 1e-5:
                continue
            correlation = np.corrcoef(qj[moving], aj[moving])[0, 1]
            if np.isfinite(correlation):
                joint_scores.append(float(correlation))
        scores.append(float(np.median(joint_scores)) if joint_scores else -1.0)
    return int(np.argmax(scores)), scores


def shift_arm_action(action: np.ndarray, correction_frames: int) -> np.ndarray:
    """Advance arm action by correction_frames; preserve gripper columns/timing."""
    if correction_frames <= 0:
        return action.copy()
    corrected = action.copy()
    corrected[:-correction_frames, :14] = action[correction_frames:, :14]
    corrected[-correction_frames:, :14] = action[-1, :14]
    return corrected


def encode_image(jpeg: bytes, quality: int) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("JPEG decode failed")
    if image.shape[:2] != (480, 640):
        image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encode failed")
    return encoded.reshape(-1)


def write_hdf5(
    path: Path, aligned: dict[str, Any], source: dict[str, Any],
    lag: int, baseline_lag: int, correction: int, quality: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.hdf5")
    if temporary.exists():
        temporary.unlink()
    action = shift_arm_action(aligned["action"], correction)
    variable_bytes = h5py.vlen_dtype(np.dtype("uint8"))
    with h5py.File(temporary, "w") as root:
        root.attrs.update({
            "sim": False,
            "compress": True,
            "training_fps": 30.0,
            "time_source": "mcap_log_time",
            "source_dataset": source["dataset"],
            "source_episode": source["episode"],
            "source_mcap": str(source["path"]),
            "estimated_arm_lag_frames": lag,
            "normal_arm_lag_frames": baseline_lag,
            "arm_lag_correction_frames": correction,
            "gripper_lag_correction_frames": 0,
            "max_sync_distance_ms": float(aligned["max_sync_distance"].max() * 1000),
            "sync_over_50ms_frames": int((aligned["max_sync_distance"] > 0.05).sum()),
        })
        root.create_dataset("action", data=action, dtype=np.float32)
        root.create_dataset("timestamp", data=aligned["times"], dtype=np.float64)
        root.create_dataset("segment_id", data=np.zeros(len(action), dtype=np.int32))
        observations = root.create_group("observations")
        observations.create_dataset("qpos", data=aligned["qpos"], dtype=np.float32)
        observations.create_dataset("qvel", data=aligned["qvel"], dtype=np.float32)
        image_group = observations.create_group("images")
        for camera_name in ("hand_left_color", "hand_right_color", "head_color"):
            dataset = image_group.create_dataset(
                camera_name, (len(action),), dtype=variable_bytes
            )
            for index, jpeg in enumerate(aligned["images"][camera_name]):
                dataset[index] = encode_image(jpeg, quality)
    temporary.replace(path)


def validate_hdf5(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as root:
        action = root["action"]
        qpos = root["observations/qpos"]
        qvel = root["observations/qvel"]
        length = len(action)
        if action.shape != (length, 16) or qpos.shape != (length, 16) or qvel.shape != (length, 16):
            raise ValueError(f"invalid numeric shapes in {path}")
        if not np.all(np.isfinite(action[:])) or not np.all(np.isfinite(qpos[:])):
            raise ValueError(f"NaN/Inf in {path}")
        camera_shapes = {}
        for camera in ("hand_left_color", "hand_right_color", "head_color"):
            dataset = root[f"observations/images/{camera}"]
            if len(dataset) != length:
                raise ValueError(f"invalid {camera} length in {path}")
            for index in sorted(set([0, length // 2, length - 1])):
                image = cv2.imdecode(np.asarray(dataset[index], dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None or image.shape != (480, 640, 3):
                    raise ValueError(f"bad {camera}[{index}] in {path}")
            camera_shapes[camera] = [length, 480, 640, 3]
        return {"frames": length, "camera_shapes": camera_shapes}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--max-lag", type=int, default=15)
    parser.add_argument("--abnormal-offset", type=int, default=3,
                        help="correct when lag exceeds median by at least this many frames")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    audit = json.loads(args.audit_json.read_text())
    ready = [item for item in audit if item["classification"] == "READY"]
    if len(ready) != 86:
        raise RuntimeError(f"expected 86 READY episodes, got {len(ready)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    lag_rows = []
    for index, source in enumerate(ready):
        print(f"[{index + 1:02d}/86] align {source['dataset']}/{source['episode']}", flush=True)
        raw = read_episode(Path(source["path"]))
        aligned = align_episode(raw)
        lag, scores = estimate_arm_lag(aligned["qpos"], aligned["action"], args.max_lag)
        lag_rows.append({
            "index": index, "dataset": source["dataset"], "episode": source["episode"],
            "estimated_lag_frames": lag, "best_score": max(scores), "scores": scores,
            "frames": len(aligned["times"]),
        })
        print(f"  frames={len(aligned['times'])}, lag={lag}, score={max(scores):.3f}", flush=True)
        del raw, aligned

    reliable_lags = [
        row["estimated_lag_frames"] for row in lag_rows if row["best_score"] >= 0.15
    ]
    if len(reliable_lags) < len(ready) * 0.75:
        raise RuntimeError(f"only {len(reliable_lags)}/86 reliable lag estimates")
    baseline_lag = int(round(statistics.median(reliable_lags)))
    print(f"Normal physical arm lag: {baseline_lag} frames", flush=True)

    manifest = []
    for index, (source, lag_row) in enumerate(zip(ready, lag_rows)):
        raw = read_episode(Path(source["path"]))
        aligned = align_episode(raw)
        lag = lag_row["estimated_lag_frames"]
        reliable = lag_row["best_score"] >= 0.15
        correction = (
            lag - baseline_lag
            if reliable and lag >= baseline_lag + args.abnormal_offset else 0
        )
        output = args.output_dir / f"episode_{index}.hdf5"
        if output.exists() and not args.overwrite:
            print(f"[{index + 1:02d}/86] exists {output.name}", flush=True)
        else:
            print(
                f"[{index + 1:02d}/86] write {output.name}: lag={lag}, correction={correction}",
                flush=True,
            )
            write_hdf5(
                output, aligned, source, lag, baseline_lag, correction, args.jpeg_quality
            )
        validation = validate_hdf5(output)
        manifest.append({
            **lag_row,
            "reliable_lag": reliable,
            "normal_lag_frames": baseline_lag,
            "arm_correction_frames": correction,
            "gripper_correction_frames": 0,
            "output": str(output),
            "max_sync_distance_ms": float(aligned["max_sync_distance"].max() * 1000),
            "sync_over_50ms_frames": int((aligned["max_sync_distance"] > 0.05).sum()),
            **validation,
        })
        del raw, aligned

    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    columns = [key for key in manifest[0] if key not in ("scores", "camera_shapes")]
    with (args.output_dir / "manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in manifest:
            writer.writerow({key: row[key] for key in columns})
    summary = {
        "episodes": len(manifest),
        "frames": sum(row["frames"] for row in manifest),
        "hours_at_30hz": sum(row["frames"] for row in manifest) / 30 / 3600,
        "normal_arm_lag_frames": baseline_lag,
        "corrected_episodes": sum(row["arm_correction_frames"] > 0 for row in manifest),
        "unreliable_lag_episodes": sum(not row["reliable_lag"] for row in manifest),
        "gripper_correction_frames": 0,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
