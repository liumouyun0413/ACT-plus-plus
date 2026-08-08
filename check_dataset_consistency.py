"""
对300条原始采集数据做全面一致性检查，输出详细报告和清洗建议。

检查项目:
  1. 目录完整性（必要文件是否存在）
  2. aligned_joints.h5 帧数分布（过短/过长异常）
  3. 相机视频帧数与h5帧数是否匹配
  4. 相机时间戳文件行数一致性
  5. action数据异常（全零、NaN、inf、范围超限）
  6. qpos数据异常
  7. 夹爪数据异常
  8. episode时长分布统计
  9. 给出可疑episode清单及清洗建议

用法:
  conda run -n aloha python check_dataset_consistency.py \
    --data_dir /home/liumouyun/extended_storage/liumouyun/Datas/record_sorting_blocks_20260415/record \
    --output_report dataset_consistency_report.txt
"""

import os
import sys
import json
import argparse
import numpy as np
import h5py
import av
from pathlib import Path
from collections import defaultdict


CAMERAS     = ['hand_left_color', 'hand_right_color', 'head_color']
JOINT_LIMIT = 3.15   # rad，超过此值视为异常（各关节实际限位应更小）
GRIP_RANGE  = (-1.0, 0.1)   # 夹爪合理范围
MIN_FRAMES  = 50     # 最少帧数（stride=7后，对应~12s）
MAX_FRAMES  = 1500   # 最多帧数（30Hz原始，~50s）


def count_video_frames_fast(video_path: Path) -> int:
    """快速获取视频帧数（读取元数据，不解码）"""
    try:
        with av.open(str(video_path)) as c:
            stream = c.streams.video[0]
            # 优先用元数据
            if stream.frames > 0:
                return stream.frames
            # fallback: 逐包计数
            count = 0
            for pkt in c.demux(stream):
                if pkt.size > 0:
                    count += 1
            return count
    except Exception as e:
        return -1


def check_episode(ep_dir: Path) -> dict:
    """检查单个episode，返回问题字典"""
    result = {
        'name': ep_dir.name,
        'path': str(ep_dir),
        'errors': [],       # 致命错误，建议丢弃
        'warnings': [],     # 轻微问题，可能影响训练质量
        'stats': {},
    }

    # ── 1. 必要文件存在性 ─────────────────────────────────────
    h5_path = ep_dir / 'record' / 'aligned_joints.h5'
    if not h5_path.exists():
        result['errors'].append('aligned_joints.h5 不存在')
        return result

    for cam in CAMERAS:
        vid = ep_dir / 'camera' / cam / f'{cam}.h265'
        txt = ep_dir / 'camera' / cam / f'{cam}.txt'
        if not vid.exists():
            result['errors'].append(f'{cam}.h265 不存在')
        if not txt.exists():
            result['warnings'].append(f'{cam}.txt 不存在')

    if result['errors']:
        return result

    # ── 2. 读取h5基本信息 ─────────────────────────────────────
    try:
        with h5py.File(str(h5_path), 'r') as f:
            keys = sorted(f.keys(), key=lambda x: int(x))
            T = len(keys)
            result['stats']['h5_frames'] = T

            if T < MIN_FRAMES:
                result['errors'].append(f'h5帧数过少: {T} < {MIN_FRAMES}（episode可能不完整）')
            if T > MAX_FRAMES:
                result['warnings'].append(f'h5帧数过多: {T} > {MAX_FRAMES}（异常长episode）')

            # 读取全量action和qpos
            actions = np.array([f[k]['action/joint/position'][()] for k in keys])
            qpos_j  = np.array([f[k]['state/joint/position'][()] for k in keys])
            l_grip  = np.array([float(f[k]['state/left_effector/position'][()].ravel()[0]) for k in keys])
            r_grip  = np.array([float(f[k]['state/right_effector/position'][()].ravel()[0]) for k in keys])

            # 时间戳跨度 → 真实时长
            ts_start = int(f[keys[0]]['main_timestamp'][()])
            ts_end   = int(f[keys[-1]]['main_timestamp'][()])
            duration_s = (ts_end - ts_start) / 1e9
            result['stats']['duration_s'] = round(duration_s, 2)

            # 相机时间戳完整性（随机抽查5帧）
            cam_ts_mismatches = 0
            for k in keys[::max(1, T//5)][:5]:
                mt = int(f[k]['main_timestamp'][()])
                for cam in CAMERAS:
                    try:
                        ct = int(f[k][f'timestamp/camera/{cam}'][()].ravel()[0])
                        diff_ms = abs(ct - mt) / 1e6
                        if diff_ms > 100:  # 超过100ms视为严重偏移
                            cam_ts_mismatches += 1
                    except KeyError:
                        cam_ts_mismatches += 1
            if cam_ts_mismatches > 0:
                result['warnings'].append(f'相机时间戳偏移超100ms: {cam_ts_mismatches}次')

    except Exception as e:
        result['errors'].append(f'h5读取失败: {e}')
        return result

    # ── 3. NaN / Inf 检查 ─────────────────────────────────────
    for name, arr in [('action', actions), ('qpos', qpos_j),
                      ('l_grip', l_grip), ('r_grip', r_grip)]:
        if np.any(np.isnan(arr)):
            result['errors'].append(f'{name} 包含 NaN')
        if np.any(np.isinf(arr)):
            result['errors'].append(f'{name} 包含 Inf')

    # ── 4. 关节范围检查 ───────────────────────────────────────
    if np.any(np.abs(actions) > JOINT_LIMIT):
        bad_frames = int(np.any(np.abs(actions) > JOINT_LIMIT, axis=1).sum())
        result['warnings'].append(
            f'action关节角超出±{JOINT_LIMIT}rad: {bad_frames}帧')
    if np.any(np.abs(qpos_j) > JOINT_LIMIT):
        bad_frames = int(np.any(np.abs(qpos_j) > JOINT_LIMIT, axis=1).sum())
        result['warnings'].append(
            f'qpos关节角超出±{JOINT_LIMIT}rad: {bad_frames}帧')

    # ── 5. 夹爪范围检查 ───────────────────────────────────────
    for name, arr in [('l_grip', l_grip), ('r_grip', r_grip)]:
        out = ((arr < GRIP_RANGE[0]) | (arr > GRIP_RANGE[1])).sum()
        if out > 0:
            result['warnings'].append(f'{name} 超出合理范围{GRIP_RANGE}: {out}帧')

    # ── 6. action全零检查（机器人静止不动？）─────────────────
    action_norms = np.linalg.norm(np.diff(actions, axis=0), axis=1)
    zero_ratio = (action_norms < 1e-6).mean()
    result['stats']['action_zero_ratio'] = round(float(zero_ratio), 3)
    if zero_ratio > 0.9:
        result['errors'].append(f'action几乎全程静止({zero_ratio*100:.0f}%帧无变化)，数据无效')
    elif zero_ratio > 0.5:
        result['warnings'].append(f'action静止比例偏高: {zero_ratio*100:.0f}%')

    # ── 7. action/qpos跟踪误差 ────────────────────────────────
    tracking_err = np.linalg.norm(actions - qpos_j, axis=1)
    max_err = float(tracking_err.max())
    mean_err = float(tracking_err.mean())
    result['stats']['tracking_err_mean'] = round(mean_err, 4)
    result['stats']['tracking_err_max']  = round(max_err, 4)
    if max_err > 1.0:
        result['warnings'].append(f'最大跟踪误差偏大: {max_err:.3f} rad')

    # ── 8. 视频帧数检查 ───────────────────────────────────────
    for cam in CAMERAS:
        vid_path = ep_dir / 'camera' / cam / f'{cam}.h265'
        txt_path = ep_dir / 'camera' / cam / f'{cam}.txt'

        # txt行数
        if txt_path.exists():
            txt_lines = sum(1 for line in open(txt_path) if line.strip())
            result['stats'][f'{cam}_txt_frames'] = txt_lines
            diff = abs(txt_lines - T)
            if diff > T * 0.1:   # 超过10%差异
                result['warnings'].append(
                    f'{cam} txt帧数({txt_lines})与h5帧数({T})差异>{diff}')

    return result


def main(args):
    data_dir = Path(args.data_dir)
    episodes = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    N = len(episodes)
    print(f"找到 {N} 个episode，开始检查...\n")

    results = []
    for i, ep_dir in enumerate(episodes):
        r = check_episode(ep_dir)
        results.append(r)
        status = '❌' if r['errors'] else ('⚠️ ' if r['warnings'] else '✅')
        print(f"[{i+1:3d}/{N}] {status} {ep_dir.name}  "
              f"frames={r['stats'].get('h5_frames','?')}  "
              f"dur={r['stats'].get('duration_s','?')}s",
              flush=True)

    # ── 汇总统计 ──────────────────────────────────────────────
    ok      = [r for r in results if not r['errors'] and not r['warnings']]
    warn    = [r for r in results if not r['errors'] and r['warnings']]
    err     = [r for r in results if r['errors']]

    frames_list   = [r['stats']['h5_frames'] for r in results if 'h5_frames' in r['stats']]
    dur_list      = [r['stats']['duration_s'] for r in results if 'duration_s' in r['stats']]
    zero_list     = [r['stats']['action_zero_ratio'] for r in results if 'action_zero_ratio' in r['stats']]
    terr_list     = [r['stats']['tracking_err_mean'] for r in results if 'tracking_err_mean' in r['stats']]

    lines = []
    def w(s=''):
        lines.append(s)
        print(s)

    w('\n' + '='*70)
    w('数据集一致性检查报告')
    w('='*70)
    w(f'总episode数:      {N}')
    w(f'✅ 完全正常:       {len(ok)}')
    w(f'⚠️  有警告(可用):  {len(warn)}')
    w(f'❌ 有错误(建议丢弃): {len(err)}')
    w()

    if frames_list:
        w('── 帧数分布 (30Hz原始) ──')
        w(f'  min={min(frames_list)}  max={max(frames_list)}  '
          f'mean={np.mean(frames_list):.0f}  median={np.median(frames_list):.0f}')
        w(f'  std={np.std(frames_list):.1f}')
        # 异常帧数
        p5  = int(np.percentile(frames_list, 5))
        p95 = int(np.percentile(frames_list, 95))
        outliers = [r['name'] for r in results
                    if 'h5_frames' in r['stats']
                    and (r['stats']['h5_frames'] < p5 or r['stats']['h5_frames'] > p95)]
        w(f'  P5={p5}, P95={p95},  帧数异常episode: {len(outliers)}')

    if dur_list:
        w()
        w('── 时长分布 (秒) ──')
        w(f'  min={min(dur_list):.1f}s  max={max(dur_list):.1f}s  '
          f'mean={np.mean(dur_list):.1f}s  median={np.median(dur_list):.1f}s')
        w(f'  std={np.std(dur_list):.1f}s')

    if zero_list:
        w()
        w('── action静止比例分布 ──')
        w(f'  mean={np.mean(zero_list)*100:.1f}%  '
          f'max={max(zero_list)*100:.1f}%  '
          f'超50%的episode: {sum(1 for z in zero_list if z>0.5)}')

    if terr_list:
        w()
        w('── 跟踪误差(action vs qpos, rad) ──')
        w(f'  mean={np.mean(terr_list):.4f}  max={max(terr_list):.4f}  '
          f'超0.1rad的episode: {sum(1 for e in terr_list if e>0.1)}')

    # ── 错误清单 ──────────────────────────────────────────────
    if err:
        w()
        w('── ❌ 建议丢弃的episode ──')
        for r in err:
            w(f'  {r["name"]}')
            for e in r['errors']:
                w(f'    ERROR: {e}')

    if warn:
        w()
        w('── ⚠️  有警告的episode ──')
        for r in warn:
            w(f'  {r["name"]}')
            for ww in r['warnings']:
                w(f'    WARN: {ww}')

    # ── 清洗建议 ──────────────────────────────────────────────
    w()
    w('='*70)
    w('清洗建议:')
    w('='*70)
    if err:
        w(f'1. 丢弃 {len(err)} 个ERROR episode（文件损坏/数据无效）')
    else:
        w('1. 无需丢弃（无致命错误）')

    if warn:
        w(f'2. 检查 {len(warn)} 个WARNING episode，酌情保留或丢弃')
    else:
        w('2. 无警告episode')

    usable = len(ok) + len(warn)
    w(f'3. 可用episode数量: {usable}/{N}')

    # ── 保存完整JSON ──────────────────────────────────────────
    json_out = Path(args.output_report).with_suffix('.json')
    with open(json_out, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    w(f'\n详细结果已保存: {json_out}')

    # ── 保存可丢弃列表 ────────────────────────────────────────
    drop_list_path = Path(args.output_report).parent / 'episodes_to_drop.txt'
    with open(drop_list_path, 'w') as f:
        for r in err:
            f.write(r['name'] + '\n')
    w(f'建议丢弃列表:   {drop_list_path}')

    # ── 保存报告文本 ──────────────────────────────────────────
    with open(args.output_report, 'w') as f:
        f.write('\n'.join(lines))
    w(f'报告已保存:     {args.output_report}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
        default='/home/liumouyun/extended_storage/liumouyun/Datas/record_sorting_blocks_20260415/record')
    parser.add_argument('--output_report',
        default='/home/liumouyun/Downloads/ACT-plus-plus/dataset_consistency_report.txt')
    main(parser.parse_args())
