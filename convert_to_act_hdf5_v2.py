"""
将采集数据转换为 ACT 训练 HDF5 格式 —— V2 (裁剪 + 填补版)
======================================================================
相对 v1 的增强:
  1. 自动裁剪首尾缺 action 的段（遥操启动前 / 结束后）
  2. 中段短缺 action 帧用前一帧 hold 填补（便于 0417 之后的数据）
  3. 跟随 check_dataset_consistency_v2.py 的 drop_list 丢弃致命 episode
  4. 支持 --fixable_json：使用 v2 检查结果中已标注的 trim_start/trim_end

数据源结构同 v1:
  {episode_dir}/
    record/aligned_joints.h5
    camera/{cam}/{cam}.h265
    camera/{cam}/{cam}.txt

用法:
  # 方式1: 直接转换，内部自动检测首尾缺帧并裁剪
  python convert_to_act_hdf5_v2.py \
    --input_dir  /home/liumouyun/extended_storage/liumouyun/Datas/record_sorting_blocks_20260417/record \
    --output_dir /home/liumouyun/extended_storage/liumouyun/Datas/record_sorting_blocks_20260417/act_dataset \
    --prefix 'sorting blocks' --stride 7 \
    --drop_list ./episodes_to_drop_v2.txt --skip_existing

  # 方式2: 使用 v2 检查脚本输出的 fixable_json (推荐)
  python convert_to_act_hdf5_v2.py \
    --input_dir  /home/liumouyun/extended_storage/liumouyun/Datas/record_sorting_blocks_20260417/record \
    --output_dir /home/liumouyun/extended_storage/liumouyun/Datas/record_sorting_blocks_20260417/act_dataset \
    --prefix 'sorting blocks' --stride 7 \
    --fixable_json ./episodes_fixable_v2.json --skip_existing
"""
import os
import sys
import json
import argparse
import numpy as np
import h5py
import av
from pathlib import Path
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────
def parse_camera_timestamps(txt_path: Path) -> dict:
    """{timestamp(int): frame_index(int)}"""
    ts2idx = {}
    with open(txt_path, 'r') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                ts = int(parts[0])
                ts2idx[ts] = idx
            except ValueError:
                continue
    return ts2idx


# ─────────────────────────────────────────────────────────────────
def detect_trim(h5_path: Path) -> tuple:
    """扫描 h5 找首尾连续缺 action 的长度, 返回 (head, tail, T)"""
    with h5py.File(str(h5_path), 'r') as f:
        keys = sorted(f.keys(), key=lambda x: int(x))
        T = len(keys)
        head = 0
        while head < T and 'action/joint/position' not in f[keys[head]]:
            head += 1
        tail = 0
        while tail < T and 'action/joint/position' not in f[keys[T - 1 - tail]]:
            tail += 1
        return head, tail, T


# ─────────────────────────────────────────────────────────────────
def load_joints_with_fill(h5_path: Path, trim_start: int, trim_end: int
                          ) -> dict:
    """
    读取 [trim_start, T - trim_end) 区间的关节数据
    中段缺 action 用前一帧 hold 填补
    返回: actions(T', 14), qpos(T', 14), l_grip(T',), r_grip(T',),
           keys_used(list of h5 keys), main_ts(T',)
    """
    with h5py.File(str(h5_path), 'r') as f:
        keys = sorted(f.keys(), key=lambda x: int(x))
        T = len(keys)
        start, end = trim_start, T - trim_end
        if start >= end:
            return None
        sub_keys = keys[start:end]
        Tp = len(sub_keys)
        actions = np.zeros((Tp, 14), dtype=np.float64)
        qpos    = np.zeros((Tp, 14), dtype=np.float64)
        qvel    = np.zeros((Tp, 14), dtype=np.float64)
        l_grip  = np.zeros(Tp, dtype=np.float64)
        r_grip  = np.zeros(Tp, dtype=np.float64)
        main_ts = np.zeros(Tp, dtype=np.int64)
        last_action = None
        n_filled = 0
        for i, k in enumerate(sub_keys):
            fr = f[k]
            main_ts[i] = int(fr['main_timestamp'][()])
            # qpos 必须存在
            if 'state/joint/position' in fr:
                qpos[i] = fr['state/joint/position'][()]
            else:
                raise RuntimeError(f'[{k}] 缺 state/joint/position')
            if 'state/joint/velocity' in fr:
                qvel[i] = fr['state/joint/velocity'][()]
            if 'state/left_effector/position' in fr:
                l_grip[i] = float(
                    fr['state/left_effector/position'][()].ravel()[0])
            if 'state/right_effector/position' in fr:
                r_grip[i] = float(
                    fr['state/right_effector/position'][()].ravel()[0])
            # action: 缺则 hold 前一帧
            if 'action/joint/position' in fr:
                last_action = fr['action/joint/position'][()]
                actions[i] = last_action
            else:
                if last_action is None:
                    # 首帧缺且没有 last (trim 不充分), 用 qpos 顶替
                    last_action = qpos[i].copy()
                actions[i] = last_action
                n_filled += 1

        # 拼 16 维 (14 臂 + 2 夹爪)
        obs_qpos = np.concatenate(
            [qpos, l_grip[:, None], r_grip[:, None]], axis=1).astype(np.float32)
        obs_qvel = np.concatenate(
            [qvel, np.zeros((Tp, 2))], axis=1).astype(np.float32)
        act_full = np.concatenate(
            [actions, l_grip[:, None], r_grip[:, None]], axis=1).astype(np.float32)
        return {
            'qpos': obs_qpos, 'qvel': obs_qvel, 'action': act_full,
            'main_ts': main_ts, 'keys': sub_keys,
            'n_filled': n_filled, 'Tp': Tp,
        }


# ─────────────────────────────────────────────────────────────────
def decode_video_frames(video_path: Path, ts_list, ts2idx: dict) -> list:
    """
    按 ts_list 中每个 timestamp 取对应视频帧. 返回 BGR ndarray list.
    视频必须按顺序解码，随机访问代价大，因此一次解完再索引。
    """
    target_frames = {}
    for ts in ts_list:
        if ts in ts2idx:
            target_frames[ts2idx[ts]] = None
    max_idx = max(target_frames.keys()) if target_frames else -1

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    try:
        stream.thread_type = 'AUTO'
    except Exception:
        pass

    results = []
    for i, frame in enumerate(container.decode(stream)):
        if i in target_frames:
            img = frame.to_ndarray(format='bgr24')
            target_frames[i] = img
        if i >= max_idx:
            break
    container.close()

    # 按 ts_list 顺序组装
    for ts in ts_list:
        idx = ts2idx.get(ts)
        if idx is None or target_frames.get(idx) is None:
            # 用前一张顶替
            if results:
                results.append(results[-1])
            else:
                # 完全无帧, 用黑色
                h, w = 480, 640
                results.append(np.zeros((h, w, 3), dtype=np.uint8))
        else:
            results.append(target_frames[idx])
    return results


# ─────────────────────────────────────────────────────────────────
def convert_episode(ep_dir: Path, camera_names, output_path: Path,
                    compress: bool, stride: int,
                    trim_start: int = None, trim_end: int = None) -> bool:
    h5_path = ep_dir / 'record' / 'aligned_joints.h5'
    if not h5_path.exists():
        print('  ❌ aligned_joints.h5 缺失')
        return False

    # 自动检测裁剪长度
    if trim_start is None or trim_end is None:
        trim_start, trim_end, _T = detect_trim(h5_path)

    data = load_joints_with_fill(h5_path, trim_start, trim_end)
    if data is None:
        print(f'  ❌ 裁剪后无有效帧 (trim {trim_start}/{trim_end})')
        return False
    Tp = data['Tp']
    print(f'  裁剪 head={trim_start} tail={trim_end}  剩余={Tp}  '
          f'action hold填补={data["n_filled"]}')

    # stride 降采样
    idx_ds = np.arange(0, Tp, stride)
    qpos_ds   = data['qpos'][idx_ds]
    qvel_ds   = data['qvel'][idx_ds]
    action_ds = data['action'][idx_ds]
    ts_ds     = data['main_ts'][idx_ds]
    T_out = len(idx_ds)
    print(f'  stride={stride} → T_out={T_out}')

    # 读取相机时间戳映射, 并按 ts_ds 取帧
    cam_images = {}
    with h5py.File(str(h5_path), 'r') as f:
        for cam in camera_names:
            # 每帧取对应相机时间戳
            cam_tss = []
            for k in data['keys']:
                key = f'timestamp/camera/{cam}'
                if key in f[k]:
                    cam_tss.append(int(f[k][key][()].ravel()[0]))
                else:
                    cam_tss.append(0)
            cam_tss = np.array(cam_tss, dtype=np.int64)[idx_ds]

            txt_path = ep_dir / 'camera' / cam / f'{cam}.txt'
            vid_path = ep_dir / 'camera' / cam / f'{cam}.h265'
            if not txt_path.exists() or not vid_path.exists():
                print(f'  ⚠️ {cam} 缺文件, 全部填黑')
                h, w = (400, 640) if 'head' in cam else (1056, 1280)
                cam_images[cam] = np.zeros((T_out, h, w, 3), dtype=np.uint8)
                continue
            ts2idx = parse_camera_timestamps(txt_path)
            imgs = decode_video_frames(vid_path, cam_tss.tolist(), ts2idx)
            cam_images[cam] = np.stack(imgs).astype(np.uint8)

    # 写 HDF5
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(output_path), 'w') as fo:
        fo.attrs['sim'] = False
        fo.attrs['compress'] = compress
        fo.create_dataset('action', data=action_ds)
        obs = fo.create_group('observations')
        obs.create_dataset('qpos', data=qpos_ds)
        obs.create_dataset('qvel', data=qvel_ds)
        imgs_g = obs.create_group('images')
        for cam, arr in cam_images.items():
            if compress:
                imgs_g.create_dataset(cam, data=arr,
                                      compression='gzip', compression_opts=4)
            else:
                imgs_g.create_dataset(cam, data=arr)
    print(f'  ✅ 保存 {output_path.name}  action={action_ds.shape} '
          f'qpos={qpos_ds.shape}  imgs[0]={list(cam_images.values())[0].shape}')
    return True


# ─────────────────────────────────────────────────────────────────
def main(args):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    drop_set = set()
    if args.drop_list and Path(args.drop_list).exists():
        with open(args.drop_list) as f:
            drop_set = {ln.strip() for ln in f if ln.strip()}
        print(f'黑名单: 跳过 {len(drop_set)} 条')

    # 优先使用 fixable_json
    trim_map = {}
    if args.fixable_json and Path(args.fixable_json).exists():
        with open(args.fixable_json) as f:
            fx = json.load(f)
        for item in fx:
            trim_map[item['name']] = (item['trim_start'], item['trim_end'])
        print(f'加载 fixable_json: {len(trim_map)} 条有裁剪信息')

    # 枚举 episode
    all_eps = sorted(
        [d for d in input_dir.iterdir()
         if d.is_dir() and d.name.startswith(args.prefix)])
    episodes = [d for d in all_eps if d.name not in drop_set]
    print(f'找到 {len(all_eps)} 个 (前缀="{args.prefix}"), '
          f'跳过 {len(all_eps)-len(episodes)}, 待转换 {len(episodes)}')

    success = skipped_existing = 0
    for idx, ep in enumerate(episodes):
        out_path = output_dir / f'episode_{idx}.hdf5'
        if args.skip_existing and out_path.exists():
            print(f'[{idx+1}/{len(episodes)}] ⏭  存在: {out_path.name}')
            skipped_existing += 1
            success += 1
            continue
        print(f'\n[{idx+1}/{len(episodes)}] {ep.name}')
        ts = trim_map.get(ep.name, (None, None))
        try:
            ok = convert_episode(ep, args.cameras, out_path,
                                 compress=not args.no_compress,
                                 stride=args.stride,
                                 trim_start=ts[0], trim_end=ts[1])
            if ok:
                success += 1
        except Exception as e:
            print(f'  ❌ 异常: {e}')

    freq = 30.0 / args.stride
    print('\n' + '=' * 60)
    print(f'✅ 完成 {success}/{len(episodes)} '
          f'(已存在跳过 {skipped_existing})')
    print(f'输出: {output_dir}')
    print(f'频率: ~{freq:.1f} Hz (stride={args.stride})')
    print(f"constants.py 建议:")
    print(f"""  '<task>': {{
      'dataset_dir': '{output_dir}',
      'num_episodes': {success},
      'episode_len': <查看生成文件实际帧数>,
      'camera_names': {args.cameras},
  }}""")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_dir',  required=True, help='episode 根目录')
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--cameras', nargs='+',
        default=['hand_left_color', 'hand_right_color', 'head_color'])
    ap.add_argument('--prefix', default='sorting_block')
    ap.add_argument('--stride', type=int, default=7)
    ap.add_argument('--no_compress', action='store_true')
    ap.add_argument('--drop_list',     default=None,
                    help='v2 检查脚本生成的 episodes_to_drop_v2.txt')
    ap.add_argument('--fixable_json',  default=None,
                    help='v2 检查脚本生成的 episodes_fixable_v2.json '
                         '(含 trim_start/trim_end)')
    ap.add_argument('--skip_existing', action='store_true')
    main(ap.parse_args())
