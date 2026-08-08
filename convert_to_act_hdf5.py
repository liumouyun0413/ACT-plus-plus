"""
将采集数据转换为ACT训练所需的HDF5格式

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
频率说明（重要）:
  aligned_joints.h5 已将关节数据上采样至 30 Hz（与相机同步），
  但关节控制指令(action)实际频率为 ~4.3 Hz（间隔~230ms），
  因此每 ~7 帧 action 才真正变化一次，其余帧为 hold 重复值。

  直接用30Hz数据训练ACT会导致模型学习大量"保持不动"的冗余动作。
  推荐做法：--stride 7  降采样到真实控制频率 ~4.3 Hz。

  对应关系:
    stride=1  → 30.0 Hz，T≈971帧，chunk_size建议100（≈3.3s）
    stride=7  → ~4.3 Hz，T≈139帧，chunk_size建议 20（≈4.7s）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

数据源结构:
  {episode_dir}/
    record/aligned_joints.h5     # 关节状态/动作，按帧索引存储（30Hz）
    camera/{cam_name}/
      {cam_name}.h265            # 视频流（30fps）
      {cam_name}.txt             # 每帧时间戳，格式: "{ts} I/P"

输出HDF5结构 (ACT标准格式):
  /observations/qpos             (T, 16)  = 14关节 + 左夹爪 + 右夹爪
  /observations/qvel             (T, 16)
  /observations/images/{cam}     (T, H, W, 3) uint8
  /action                        (T, 16)

用法:
  # 推荐：降采样到控制频率 ~4.3Hz
  python convert_to_act_hdf5.py \
    --input_dir /home/liumouyun/Downloads/ACT-plus-plus \
    --output_dir /home/liumouyun/Downloads/ACT-plus-plus/act_dataset \
    --cameras hand_left_color hand_right_color head_color \
    --prefix sorting_block \
    --stride 7

  # 保持30Hz（不推荐，action大量重复）
  python convert_to_act_hdf5.py ... --stride 1
"""

import os
import sys
import argparse
import numpy as np
import h5py
import av
from pathlib import Path
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────
# 相机时间戳解析
# ─────────────────────────────────────────────────────────────────

def parse_camera_timestamps(txt_path: Path) -> dict:
    """
    解析 .txt 时间戳文件，返回 {timestamp(int): frame_index(int)}
    文件格式每行: "1776308736044918624 I"
    """
    ts_to_idx = {}
    with open(txt_path) as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            ts = int(parts[0])
            ts_to_idx[ts] = idx
    return ts_to_idx


# ─────────────────────────────────────────────────────────────────
# H265 视频解码
# ─────────────────────────────────────────────────────────────────

def decode_h265_all_frames(video_path: Path) -> list:
    """
    用 PyAV 解码整个 h265 视频，返回 list of np.ndarray (H, W, 3) uint8 RGB
    """
    frames = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        for packet in container.demux(stream):
            for frame in packet.decode():
                # frame.to_ndarray 默认是 yuv，指定 rgb24
                img = frame.to_ndarray(format='rgb24')
                frames.append(img)
    return frames


# ─────────────────────────────────────────────────────────────────
# 单个 episode 转换
# ─────────────────────────────────────────────────────────────────

def detect_action_stride(h5_path: Path, sample: int = 200) -> int:
    """
    自动检测action的真实更新间隔（stride）。
    做法：统计相邻帧action/joint/position变化量，找到典型非零间隔。
    """
    with h5py.File(str(h5_path), 'r') as f:
        keys = sorted(f.keys(), key=lambda x: int(x))[:sample]
        actions = np.array([f[k]['action/joint/position'][()] for k in keys])
    diffs = np.linalg.norm(np.diff(actions, axis=0), axis=1)
    changed = np.where(diffs > 1e-5)[0]
    if len(changed) < 2:
        return 1
    gaps = np.diff(changed)
    stride = int(np.median(gaps))
    return max(1, stride)


def convert_episode(episode_dir: Path, camera_names: list, output_path: Path,
                    compress: bool = True, stride: int = 7):
    """
    转换一个 episode 目录 → ACT HDF5 文件

    关节维度说明 (共16维):
      [0:14]  双臂14关节 (state/action/joint/position)
      [14]    左夹爪    (state/action/left_effector/position)
      [15]    右夹爪    (state/action/right_effector/position)

    stride: 降采样步长。stride=7 将30Hz数据降至~4.3Hz（真实控制频率）。
    """
    h5_path = episode_dir / 'record' / 'aligned_joints.h5'
    if not h5_path.exists():
        print(f"  ❌ 找不到 aligned_joints.h5: {h5_path}")
        return False

    # ── 0. 自动检测/验证 stride ─────────────────────────────────
    auto_stride = detect_action_stride(h5_path)
    print(f"  🔍 自动检测action更新间隔: {auto_stride} 帧 "
          f"(≈{1000/33.33/auto_stride:.1f} Hz)，使用 stride={stride}")
    if stride != auto_stride:
        print(f"  ⚠️  指定stride={stride} 与检测值{auto_stride}不同，"
              f"按指定值继续")

    # ── 1. 读取关节数据（全量，后续按stride采样）─────────────────
    print(f"  📖 读取关节数据...")
    with h5py.File(str(h5_path), 'r') as f:
        all_keys = sorted(f.keys(), key=lambda x: int(x))
        T_full = len(all_keys)

        qpos_full   = np.zeros((T_full, 16), dtype=np.float32)
        qvel_full   = np.zeros((T_full, 16), dtype=np.float32)
        action_full = np.zeros((T_full, 16), dtype=np.float32)
        cam_ts_full = {cam: np.zeros(T_full, dtype=np.int64)
                       for cam in camera_names}

        for i, k in enumerate(all_keys):
            g = f[k]
            qpos_full[i, :14]  = g['state/joint/position'][()]
            qvel_full[i, :14]  = g['state/joint/velocity'][()]
            qpos_full[i, 14]   = g['state/left_effector/position'][()].ravel()[0]
            qpos_full[i, 15]   = g['state/right_effector/position'][()].ravel()[0]

            action_full[i, :14] = g['action/joint/position'][()]
            action_full[i, 14]  = g['action/left_effector/position'][()].ravel()[0]
            action_full[i, 15]  = g['action/right_effector/position'][()].ravel()[0]

            for cam in camera_names:
                try:
                    cam_ts_full[cam][i] = int(
                        g[f'timestamp/camera/{cam}'][()].ravel()[0])
                except KeyError:
                    cam_ts_full[cam][i] = -1

    # ── 2. 按 stride 降采样关节数据 ──────────────────────────────
    indices = np.arange(0, T_full, stride)
    T = len(indices)
    qpos   = qpos_full[indices]
    qvel   = qvel_full[indices]
    action = action_full[indices]
    cam_ts_per_frame = {cam: cam_ts_full[cam][indices] for cam in camera_names}

    # 统计降采样后action真实变化率
    action_diffs = np.linalg.norm(np.diff(action, axis=0), axis=1)
    change_ratio = (action_diffs > 1e-5).mean()
    print(f"  ✅ 关节数据: {T_full}帧(30Hz) → {T}帧(stride={stride}, "
          f"~{1000/33.33/stride:.1f}Hz)")
    print(f"     降采样后action帧间变化率: {change_ratio*100:.1f}% "
          f"({'正常' if change_ratio > 0.5 else '⚠️ 仍有大量重复，考虑增大stride'})")

    # ── 3. 解码相机视频并对齐（只取降采样后的T帧）───────────────
    images = {}
    for cam in camera_names:
        cam_dir  = episode_dir / 'camera' / cam
        vid_path = cam_dir / f'{cam}.h265'
        txt_path = cam_dir / f'{cam}.txt'

        if not vid_path.exists():
            print(f"  ⚠️  找不到视频: {vid_path}, 用黑色占位")
            images[cam] = None
            continue

        print(f"  🎬 解码 {cam} ...")
        all_frames = decode_h265_all_frames(vid_path)
        print(f"     视频总帧数: {len(all_frames)}")

        # 解析时间戳映射
        if txt_path.exists():
            ts_to_vidx = parse_camera_timestamps(txt_path)
        else:
            # 无时间戳文件：直接按顺序截取
            print(f"  ⚠️  找不到时间戳文件 {txt_path}，按顺序截取")
            ts_to_vidx = {}

        H, W = all_frames[0].shape[:2]
        cam_imgs = np.zeros((T, H, W, 3), dtype=np.uint8)

        matched = 0
        for i in range(T):
            ts = cam_ts_per_frame[cam][i]
            if ts != -1 and ts in ts_to_vidx:
                vidx = ts_to_vidx[ts]
                if vidx < len(all_frames):
                    cam_imgs[i] = all_frames[vidx]
                    matched += 1
            elif ts_to_vidx:
                # 找最近时间戳
                best = min(ts_to_vidx.keys(), key=lambda t: abs(t - ts))
                vidx = ts_to_vidx[best]
                if vidx < len(all_frames):
                    cam_imgs[i] = all_frames[vidx]
                    matched += 1
            else:
                # fallback: 按比例映射
                vidx = min(i, len(all_frames) - 1)
                cam_imgs[i] = all_frames[vidx]
                matched += 1

        print(f"     对齐帧数: {matched}/{T}")
        images[cam] = cam_imgs

    # ── 3. 写入 ACT HDF5 ────────────────────────────────────────
    comp = 'lzf' if compress else None
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  💾 写入 {output_path} ...")
    with h5py.File(str(output_path), 'w', rdcc_nbytes=1024**2 * 4) as root:
        root.attrs['sim'] = False
        root.attrs['compress'] = False

        obs = root.create_group('observations')
        obs.create_dataset('qpos', data=qpos, compression=comp, dtype='float32')
        obs.create_dataset('qvel', data=qvel, compression=comp, dtype='float32')

        img_grp = obs.create_group('images')
        for cam in camera_names:
            if images.get(cam) is not None:
                img_grp.create_dataset(
                    cam, data=images[cam],
                    dtype='uint8', compression=comp,
                    chunks=(1, *images[cam].shape[1:])
                )
            else:
                # 黑色占位 480x640
                dummy = np.zeros((T, 480, 640, 3), dtype=np.uint8)
                img_grp.create_dataset(
                    cam, data=dummy,
                    dtype='uint8', compression=comp,
                    chunks=(1, 480, 640, 3)
                )

        root.create_dataset('action', data=action, compression=comp, dtype='float32')

    print(f"  ✅ 保存完成")
    return True


# ─────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────

def main(args):
    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 读取黑名单
    drop_set = set()
    if args.drop_list and Path(args.drop_list).exists():
        with open(args.drop_list) as f:
            drop_set = {line.strip() for line in f if line.strip()}
        print(f"黑名单: 跳过 {len(drop_set)} 个episode")

    # 查找所有匹配前缀的 episode 目录，过滤黑名单
    all_episodes = sorted([
        d for d in input_dir.iterdir()
        if d.is_dir() and d.name.startswith(args.prefix)
    ])
    episodes = [d for d in all_episodes if d.name not in drop_set]
    skipped  = [d for d in all_episodes if d.name in drop_set]

    print(f"找到 {len(all_episodes)} 个episode (前缀='{args.prefix}')")
    print(f"跳过黑名单: {len(skipped)} 个，待转换: {len(episodes)} 个")
    if skipped:
        for ep in skipped:
            print(f"  ⛔ {ep.name}")

    success = 0
    skipped_existing = 0
    for idx, ep_dir in enumerate(episodes):
        out_path = output_dir / f'episode_{idx}.hdf5'
        # 断点续传：已存在则跳过
        if args.skip_existing and out_path.exists():
            print(f"[{idx+1}/{len(episodes)}] ⏭  已存在，跳过: {out_path.name}")
            skipped_existing += 1
            success += 1
            continue
        print(f"\n[{idx+1}/{len(episodes)}] 处理: {ep_dir.name}")
        ok = convert_episode(ep_dir, args.cameras, out_path,
                             compress=not args.no_compress,
                             stride=args.stride)
        if ok:
            success += 1

    print(f"\n{'='*50}")
    print(f"✅ 完成: {success}/{len(episodes)} 个episode"
          + (f" (其中 {skipped_existing} 个已存在跳过)" if skipped_existing else ""))
    print(f"输出目录: {output_dir}")
    print(f"\n请将以下配置添加到 constants.py:")
    freq = 30.0 / args.stride
    print(f"""
'sorting_block': {{
    'dataset_dir': '{output_dir}',
    'num_episodes': {success},
    'episode_len': <查看实际帧数>,   # stride={args.stride} → ~{freq:.1f}Hz
    'camera_names': {args.cameras},
}},
# 训练时建议 chunk_size={int(round(freq * 5))}  (覆盖约5秒动作)
# imitate_episodes.py --chunk_size {int(round(freq * 5))}
""")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',  default='/home/liumouyun/Downloads/ACT-plus-plus',
                        help='episode根目录')
    parser.add_argument('--output_dir', default='/home/liumouyun/Downloads/ACT-plus-plus/act_dataset',
                        help='ACT HDF5输出目录')
    parser.add_argument('--cameras', nargs='+',
                        default=['hand_left_color', 'hand_right_color', 'head_color'],
                        help='要转换的相机名称')
    parser.add_argument('--prefix', default='sorting_block',
                        help='episode目录名称前缀')
    parser.add_argument('--stride', type=int, default=7,
                        help='降采样步长: 7=~4.3Hz(推荐), 1=30Hz(不推荐)')
    parser.add_argument('--no_compress', action='store_true', help='不压缩HDF5')
    parser.add_argument('--drop_list', default=None,
                        help='黑名单文件路径，每行一个episode目录名，这些episode将被跳过')
    parser.add_argument('--skip_existing', action='store_true',
                        help='跳过已存在的HDF5文件（用于断点续传）')
    main(parser.parse_args())
