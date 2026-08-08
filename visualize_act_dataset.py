"""
将 ACT HDF5 数据集渲染成可视化视频
- 多相机画面拼接（上半）
- 关节角度/夹爪曲线图（下半）

用法:
  python visualize_act_dataset.py --episode 0 --output_dir ./vis_videos
"""

import argparse
import numpy as np
import h5py
import cv2
from pathlib import Path


JOINT_LABELS = [
    'L_J1','L_J2','L_J3','L_J4','L_J5','L_J6','L_J7',
    'R_J1','R_J2','R_J3','R_J4','R_J5','R_J6','R_J7',
    'L_grip','R_grip',
]


def resize_frame(img: np.ndarray, height: int) -> np.ndarray:
    """等比缩放到目标高度"""
    h, w = img.shape[:2]
    new_w = int(w * height / h)
    return cv2.resize(img, (new_w, height), interpolation=cv2.INTER_AREA)


def draw_joint_curves(qpos: np.ndarray, action: np.ndarray,
                      current_t: int, width: int, height: int) -> np.ndarray:
    """
    绘制关节角度曲线面板
    蓝色=qpos(实际), 红色=action(指令), 竖线=当前帧
    只画左臂7关节+左夹爪(左半) / 右臂7关节+右夹爪(右半)
    """
    panel = np.ones((height, width, 3), dtype=np.uint8) * 30  # 深灰背景
    T = len(qpos)

    groups = [
        ('Left Arm + Gripper', list(range(7)) + [14], (0, 255, 100)),
        ('Right Arm + Gripper', list(range(7, 14)) + [15], (0, 180, 255)),
    ]

    half_w = width // 2
    for gi, (title, joint_ids, color) in enumerate(groups):
        x_off = gi * half_w
        n = len(joint_ids)
        cell_h = (height - 30) // n
        # 标题
        cv2.putText(panel, title, (x_off + 5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

        for ri, jid in enumerate(joint_ids):
            y_top = 30 + ri * cell_h
            y_bot = y_top + cell_h - 2
            cell_h_px = y_bot - y_top

            q_vals = qpos[:, jid]
            a_vals = action[:, jid]
            vmin = min(q_vals.min(), a_vals.min()) - 0.1
            vmax = max(q_vals.max(), a_vals.max()) + 0.1
            span = max(vmax - vmin, 0.01)

            def to_px(val):
                norm = (val - vmin) / span
                return int(y_bot - norm * cell_h_px)

            def to_x(t):
                return x_off + int(t / T * half_w)

            # 画qpos(蓝) 和 action(红)
            for series, col in [(q_vals, (180, 120, 50)), (a_vals, (50, 80, 220))]:
                pts = [(to_x(t), to_px(series[t])) for t in range(T)]
                for t in range(1, T):
                    cv2.line(panel, pts[t-1], pts[t], col, 1)

            # 当前帧竖线
            cx = to_x(current_t)
            cv2.line(panel, (cx, y_top), (cx, y_bot), (255, 255, 0), 1)

            # 标签 + 当前值
            label = f"{JOINT_LABELS[jid]}:{qpos[current_t, jid]:.2f}"
            cv2.putText(panel, label, (x_off + 3, y_top + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (200, 200, 200), 1)

    # 图例
    cv2.rectangle(panel, (width-110, height-22), (width-5, height-2), (40,40,40), -1)
    cv2.line(panel, (width-105, height-12), (width-80, height-12), (180,120,50), 2)
    cv2.putText(panel, 'qpos', (width-78, height-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180,120,50), 1)
    cv2.line(panel, (width-55, height-12), (width-30, height-12), (50,80,220), 2)
    cv2.putText(panel, 'action', (width-28, height-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (50,80,220), 1)

    return panel


def hdf5_to_video(hdf5_path: str, output_path: str,
                  cam_height: int = 360, fps: float = None):
    """
    将单个HDF5 episode渲染为视频
    上半：多相机拼接
    下半：关节曲线
    fps=None 时自动从 HDF5 attrs 读取，否则回退到 30/7 ≈ 4.286Hz
    """
    with h5py.File(hdf5_path, 'r') as f:
        cam_names = list(f['/observations/images'].keys())
        qpos   = f['/observations/qpos'][()]   # (T, 16)
        action = f['/action'][()]               # (T, 16)
        T = len(qpos)
        compressed = bool(f.attrs.get('compress', False))
        segment_ids = f['segment_id'][()] if 'segment_id' in f else np.zeros(T, dtype=np.int32)

        # 自动推断 fps：优先读 attrs，否则用 30/7 ≈ 4.286Hz（stride=7 标准）
        if fps is None:
            if 'training_fps' in f.attrs:
                fps = float(f.attrs['training_fps'])
            elif 'fps' in f.attrs:
                fps = float(f.attrs['fps'])
            elif 'stride' in f.attrs:
                fps = 30.0 / float(f.attrs['stride'])
            else:
                fps = 30.0 / 7  # 默认 stride=7 → ~4.286Hz

        print(f"  相机: {cam_names}")
        print(f"  帧数T={T}, fps={fps:.1f} → 时长{T/fps:.1f}s")

        def read_frame(cam, index):
            data = f[f'/observations/images/{cam}'][index]
            if compressed:
                img = cv2.imdecode(np.asarray(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError(f'JPEG解码失败: {cam}[{index}]')
                return img
            return cv2.cvtColor(data, cv2.COLOR_RGB2BGR)

        # 确定拼接后宽度（等比缩放各相机到cam_height）
        sample_frames = [resize_frame(read_frame(cam, 0), cam_height) for cam in cam_names]
        cam_row_w = sum(frame.shape[1] for frame in sample_frames)
        curve_h = 300
        total_h = cam_height + curve_h
        total_w = cam_row_w

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_path.replace('.mp4', '_tmp.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(tmp_path, fourcc, fps, (total_w, total_h))
        if not writer.isOpened():
            raise RuntimeError(f"cv2.VideoWriter 无法打开: {tmp_path}")

        print(f"  渲染 {total_w}x{total_h} @ {fps}fps ...")

        for t in range(T):
            # ── 相机行 ────────────────────────────────
            frames = []
            for cam in cam_names:
                img_bgr = read_frame(cam, t)
                img_rsz = resize_frame(img_bgr, cam_height)
                label = cam.replace('_color','').replace('_',' ')
                cv2.putText(img_rsz, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 0, 0), 3)
                cv2.putText(img_rsz, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (255, 255, 255), 1)
                frames.append(img_rsz)

            cam_row = np.concatenate(frames, axis=1)

        # ── 曲线面板 ──────────────────────────────
            curve = draw_joint_curves(qpos, action, t, total_w, curve_h)

        # ── 合并 ──────────────────────────────────
            combined = np.concatenate([cam_row, curve], axis=0)

        # 帧号 + 时间戳
            info = f"t={t:04d}/{T}  {t/fps:.1f}s  segment={segment_ids[t]}"
            cv2.putText(combined, info, (total_w - 330, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3)
            cv2.putText(combined, info, (total_w - 330, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,100), 1)

            writer.write(combined)

            if t % 100 == 0:
                print(f"    {t}/{T}", end='\r', flush=True)

        writer.release()
    # 用 ffmpeg 转为 H264（VS Code 可直接播放）。输出到独立临时文件，
    # 防止中断时留下只有 MP4 header 的伪成品。
    import subprocess, os
    transcode_path = output_path.replace('.mp4', '_transcoding.mp4')
    if os.path.exists(transcode_path):
        os.remove(transcode_path)
    command = [
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
        '-i', tmp_path, '-vcodec', 'libx264', '-preset', 'veryfast',
        '-crf', '23', '-movflags', '+faststart', transcode_path,
    ]
    try:
        ret = subprocess.run(command, capture_output=True, timeout=300)
    except subprocess.TimeoutExpired:
        ret = None
        print('  ⚠️ ffmpeg 转换超过 300 秒，保留 mp4v 版本')
    if ret is None or ret.returncode != 0:
        if ret is not None:
            print(f"  ⚠️ ffmpeg 转换失败: {ret.stderr.decode(errors='replace')[-500:]}")
        if os.path.exists(transcode_path):
            os.remove(transcode_path)
        os.replace(tmp_path, output_path)
    else:
        os.replace(transcode_path, output_path)
        os.remove(tmp_path)
    print(f"\n  ✅ 保存: {output_path}")


def main(args):
    dataset_dir = Path(args.dataset_dir)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.episode == -1:
        hdf5_files = sorted(dataset_dir.glob('episode_*.hdf5'),
                            key=lambda path: int(path.stem.split('_')[-1]))
    else:
        hdf5_files = [dataset_dir / f'episode_{args.episode}.hdf5']

    for hdf5_path in hdf5_files:
        if not hdf5_path.exists():
            print(f"❌ 找不到: {hdf5_path}")
            continue
        out_path = output_dir / (hdf5_path.stem + '.mp4')
        if out_path.exists() and out_path.stat().st_size > 0 and not args.overwrite:
            print(f"⏭️ 已存在，跳过: {out_path}")
            continue
        print(f"\n▶ {hdf5_path.name}")
        hdf5_to_video(str(hdf5_path), str(out_path),
                      cam_height=args.cam_height,
                      fps=args.fps)

    print(f"\n全部完成，视频保存在: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default='act_dataset')
    parser.add_argument('--output_dir',  default='vis_videos')
    parser.add_argument('--episode', type=int, default=0,
                        help='episode编号，-1表示全部')
    parser.add_argument('--cam_height', type=int, default=360,
                        help='相机画面缩放高度(像素)')
    parser.add_argument('--fps', type=float, default=None,
                        help='输出视频帧率（默认自动：30/stride≈4.3Hz；慢放可指定10）')
    parser.add_argument('--overwrite', action='store_true',
                        help='覆盖已经生成的视频（默认断点续传）')
    main(parser.parse_args())
