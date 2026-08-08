"""
数据集一致性检查 V2（增强版）
======================================================================
相对 v1 的增强:
  * 所有字段读取改为"安全读取"，缺失字段不会崩溃
  * 新增分段统计: 首段缺action帧数 / 末段缺action帧数 / 中段最大连续缺帧
  * 新增左/右臂 qpos 连续重复比例 (疑似单侧丢帧被 hold)
  * 新增 main_timestamp 单调/跳变检查
  * 新增相机时间戳全帧扫描 (缺失 / =0 / >100ms 偏移分别统计)

判定规则:
  - ERROR (建议丢弃):
      * 必要文件不存在
      * h5 帧数过少
      * qpos/夹爪 缺失 > 2%
      * 中段连续缺 action > 30 帧
      * 相机时间戳字段缺失 / 为0 比例 > 2%
      * main_timestamp 非单调
  - WARNING (可修复, convert 时裁剪+填补即可):
      * 首段或末段缺 action (任意长度)
      * 中段缺 action 比例 < 20% 且最大连续段 ≤ 30 帧
      * 左右臂卡住比例差 > 30%
      * 跟踪误差/关节角超限等

用法:
  conda run -n aloha python check_dataset_consistency_v2.py \
    --data_dir /home/liumouyun/extended_storage/liumouyun/Datas/record_sorting_blocks_20260417/record \
    --output_report ./dataset_consistency_report_0417_v2.txt
"""

import os
import json
import argparse
import numpy as np
import h5py
from pathlib import Path
from collections import defaultdict


CAMERAS     = ['hand_left_color', 'hand_right_color', 'head_color']
JOINT_LIMIT = 3.15
GRIP_RANGE  = (-1.0, 0.1)
MIN_FRAMES  = 50
MAX_FRAMES  = 1500

# 缺 action 判定阈值
MIDDLE_GAP_FATAL   = 30       # 中段连续缺超过此值 → ERROR
MIDDLE_MISS_RATIO_FATAL = 0.20  # 中段缺帧比例超过此值 → ERROR
FIELD_MISS_RATIO_FATAL  = 0.02  # qpos/夹爪/相机时间戳缺失比例阈值


def segment_missing(miss_indices, T, head_ratio=0.1, tail_ratio=0.1):
    """
    将缺失帧索引切分为 head / middle / tail 三段，返回:
      head_count, tail_count, middle_count, middle_max_gap
    head = 从第0帧开始连续缺的长度
    tail = 从最后一帧倒数连续缺的长度
    middle = 余下的缺帧数量，middle_max_gap = 中段里最长一段连续缺帧长度
    """
    miss_set = set(miss_indices)

    # head: 从 0 开始连续
    head = 0
    while head < T and head in miss_set:
        head += 1
    # tail: 从 T-1 倒推连续
    tail = 0
    while tail < T and (T - 1 - tail) in miss_set:
        tail += 1

    middle_miss = sorted(i for i in miss_indices if head <= i < T - tail)
    middle_count = len(middle_miss)

    middle_max_gap = 0
    if middle_miss:
        cur = 1
        for i in range(1, len(middle_miss)):
            if middle_miss[i] == middle_miss[i - 1] + 1:
                cur += 1
                middle_max_gap = max(middle_max_gap, cur)
            else:
                cur = 1
        middle_max_gap = max(middle_max_gap, cur)

    return head, tail, middle_count, middle_max_gap


def check_episode(ep_dir: Path) -> dict:
    result = {
        'name': ep_dir.name,
        'path': str(ep_dir),
        'errors': [],
        'warnings': [],
        'stats': {},
    }

    # ── 1. 必要文件 ──────────────────────────────────────────
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

    # ── 2. 安全读取 h5 ────────────────────────────────────────
    try:
        with h5py.File(str(h5_path), 'r') as f:
            keys = sorted(f.keys(), key=lambda x: int(x))
            T = len(keys)
            result['stats']['h5_frames'] = T

            if T < MIN_FRAMES:
                result['errors'].append(
                    f'h5帧数过少: {T} < {MIN_FRAMES}')
                return result
            if T > MAX_FRAMES:
                result['warnings'].append(
                    f'h5帧数过多: {T} > {MAX_FRAMES}')

            actions = np.zeros((T, 14), dtype=np.float64)
            qpos_j  = np.zeros((T, 14), dtype=np.float64)
            l_grip  = np.zeros(T, dtype=np.float64)
            r_grip  = np.zeros(T, dtype=np.float64)
            miss_action = []
            miss_qpos   = []
            miss_lgrip  = []
            miss_rgrip  = []

            for i, k in enumerate(keys):
                fr = f[k]
                if 'action/joint/position' in fr:
                    a = fr['action/joint/position'][()]
                    if a.shape == (14,):
                        actions[i] = a
                    else:
                        miss_action.append(i)
                else:
                    miss_action.append(i)
                if 'state/joint/position' in fr:
                    q = fr['state/joint/position'][()]
                    if q.shape == (14,):
                        qpos_j[i] = q
                    else:
                        miss_qpos.append(i)
                else:
                    miss_qpos.append(i)
                if 'state/left_effector/position' in fr:
                    l_grip[i] = float(
                        fr['state/left_effector/position'][()].ravel()[0])
                else:
                    miss_lgrip.append(i)
                if 'state/right_effector/position' in fr:
                    r_grip[i] = float(
                        fr['state/right_effector/position'][()].ravel()[0])
                else:
                    miss_rgrip.append(i)

            # ── 2.1 action 分段统计 ────────────────────────────
            h, t, m, mx = segment_missing(miss_action, T)
            result['stats']['miss_action_total']       = len(miss_action)
            result['stats']['miss_action_head']        = h
            result['stats']['miss_action_tail']        = t
            result['stats']['miss_action_middle']      = m
            result['stats']['miss_action_middle_max']  = mx
            result['stats']['usable_range']            = [h, T - t]  # [start, end)
            result['stats']['usable_frames']           = max(0, T - h - t)

            if mx > MIDDLE_GAP_FATAL:
                result['errors'].append(
                    f'中段连续缺 action 过长: {mx} > {MIDDLE_GAP_FATAL}')
            usable = max(1, T - h - t)
            if m / usable > MIDDLE_MISS_RATIO_FATAL:
                result['errors'].append(
                    f'中段 action 缺帧比例过高: {m}/{usable}='
                    f'{m/usable*100:.1f}% > {MIDDLE_MISS_RATIO_FATAL*100:.0f}%')
            if h > 0 or t > 0:
                result['warnings'].append(
                    f'首/末段缺 action: head={h} tail={t} '
                    f'(建议 convert 时裁剪至 [{h},{T-t}))')
            if 0 < m <= int(MIDDLE_MISS_RATIO_FATAL * usable) and mx <= MIDDLE_GAP_FATAL:
                result['warnings'].append(
                    f'中段 action 零星缺 {m} 帧 (最长连续 {mx}), convert 时 hold 填补即可')

            # ── 2.2 qpos / 夹爪字段缺失 ────────────────────────
            for name, miss, label in [
                ('qpos', miss_qpos,  'qpos'),
                ('left_effector', miss_lgrip,  '左夹爪'),
                ('right_effector', miss_rgrip, '右夹爪'),
            ]:
                result['stats'][f'miss_{name}_frames'] = len(miss)
                if miss:
                    tag = ('errors' if len(miss) / T > FIELD_MISS_RATIO_FATAL
                           else 'warnings')
                    result[tag].append(f'{label} 字段缺失 {len(miss)}/{T} 帧')

            # ── 2.3 NaN / Inf ─────────────────────────────────
            # 仅对有效区间 [h, T-t) 检查
            if h < T - t:
                a_valid = actions[h:T - t]
                q_valid = qpos_j[h:T - t]
                lg_valid = l_grip[h:T - t]
                rg_valid = r_grip[h:T - t]
                for nm, arr in [('action', a_valid), ('qpos', q_valid),
                                ('l_grip', lg_valid), ('r_grip', rg_valid)]:
                    if np.any(np.isnan(arr)):
                        result['errors'].append(f'{nm} 包含 NaN')
                    if np.any(np.isinf(arr)):
                        result['errors'].append(f'{nm} 包含 Inf')

                # ── 2.4 关节范围 ────────────────────────────────
                if np.any(np.abs(a_valid) > JOINT_LIMIT):
                    bad = int(np.any(np.abs(a_valid) > JOINT_LIMIT, axis=1).sum())
                    result['warnings'].append(
                        f'action关节角超 ±{JOINT_LIMIT}rad: {bad}帧')
                if np.any(np.abs(q_valid) > JOINT_LIMIT):
                    bad = int(np.any(np.abs(q_valid) > JOINT_LIMIT, axis=1).sum())
                    result['warnings'].append(
                        f'qpos关节角超 ±{JOINT_LIMIT}rad: {bad}帧')

                # ── 2.5 夹爪范围 ────────────────────────────────
                for nm, arr in [('l_grip', lg_valid), ('r_grip', rg_valid)]:
                    out = int(((arr < GRIP_RANGE[0]) | (arr > GRIP_RANGE[1])).sum())
                    if out > 0:
                        result['warnings'].append(
                            f'{nm} 超出 {GRIP_RANGE}: {out}帧')

                # ── 2.6 左右臂卡住比例（整段 qpos，不受缺action影响） ─
                Tv = len(q_valid)
                if Tv > 10:
                    stk_l = int(np.all(np.diff(q_valid[:, :7], axis=0) == 0,
                                       axis=1).sum())
                    stk_r = int(np.all(np.diff(q_valid[:, 7:14], axis=0) == 0,
                                       axis=1).sum())
                    rl = round(stk_l / (Tv - 1), 3)
                    rr = round(stk_r / (Tv - 1), 3)
                    result['stats']['stuck_qpos_left_ratio']  = rl
                    result['stats']['stuck_qpos_right_ratio'] = rr
                    if abs(rl - rr) > 0.3:
                        result['warnings'].append(
                            f'qpos 左右卡住差异大 L={rl:.2f} R={rr:.2f} '
                            '(疑似单侧丢帧)')
                    for side, v in [('左臂', rl), ('右臂', rr)]:
                        if v > 0.5:
                            result['warnings'].append(
                                f'qpos {side}卡住 {v:.2f} > 0.5 (疑似丢帧被hold)')

                # ── 2.7 跟踪误差 (仅比较有 action 的帧) ──────────
                valid_mask = np.ones(T - t - h, dtype=bool)
                for i in miss_action:
                    j = i - h
                    if 0 <= j < T - t - h:
                        valid_mask[j] = False
                if valid_mask.any():
                    err = np.linalg.norm(
                        a_valid[valid_mask] - q_valid[valid_mask], axis=1)
                    result['stats']['tracking_err_mean'] = round(float(err.mean()), 4)
                    result['stats']['tracking_err_max']  = round(float(err.max()), 4)
                    if err.max() > 1.0:
                        result['warnings'].append(
                            f'最大跟踪误差 {err.max():.3f} rad')

                # ── 2.8 action 全零 (有效帧内) ───────────────────
                if valid_mask.sum() > 1:
                    a_v = a_valid[valid_mask]
                    norms = np.linalg.norm(np.diff(a_v, axis=0), axis=1)
                    zr = float((norms < 1e-6).mean())
                    result['stats']['action_zero_ratio'] = round(zr, 3)
                    if zr > 0.9:
                        result['errors'].append(
                            f'action 几乎全程静止 {zr*100:.0f}%')
                    elif zr > 0.5:
                        result['warnings'].append(
                            f'action 静止比例偏高 {zr*100:.0f}%')

            # ── 2.9 main_timestamp ────────────────────────────
            ts_all = np.array(
                [int(f[k]['main_timestamp'][()]) for k in keys], dtype=np.int64)
            result['stats']['duration_s'] = round(
                float(ts_all[-1] - ts_all[0]) / 1e9, 2)
            dts = np.diff(ts_all) / 1e6
            n_nonmono = int((dts <= 0).sum())
            if n_nonmono > 0:
                result['errors'].append(
                    f'main_timestamp 非单调递增: {n_nonmono}处')
            if len(dts) > 0:
                med = float(np.median(dts))
                big = int((dts > med * 2.5).sum())
                result['stats']['ts_median_ms'] = round(med, 2)
                result['stats']['ts_big_gaps']  = big
                if big > 0:
                    result['warnings'].append(
                        f'main_timestamp 跳变>2.5×中位: {big}处 '
                        f'(最大 {float(dts.max()):.0f}ms)')

            # ── 2.10 相机时间戳全帧扫描 ────────────────────────
            cam_missing = defaultdict(int)
            cam_zero    = defaultdict(int)
            cam_big_off = defaultdict(int)
            for k in keys:
                mt = int(f[k]['main_timestamp'][()])
                for cam in CAMERAS:
                    key = f'timestamp/camera/{cam}'
                    if key not in f[k]:
                        cam_missing[cam] += 1
                        continue
                    ct = int(f[k][key][()].ravel()[0])
                    if ct == 0:
                        cam_zero[cam] += 1
                        continue
                    if abs(ct - mt) / 1e6 > 100:
                        cam_big_off[cam] += 1
            for cam in CAMERAS:
                result['stats'][f'{cam}_missing_ts'] = cam_missing[cam]
                result['stats'][f'{cam}_zero_ts']    = cam_zero[cam]
                result['stats'][f'{cam}_big_off_ts'] = cam_big_off[cam]
                if cam_missing[cam]:
                    result['errors'].append(
                        f'{cam} 时间戳字段缺失 {cam_missing[cam]}/{T}')
                if cam_zero[cam]:
                    tag = ('errors' if cam_zero[cam] / T > FIELD_MISS_RATIO_FATAL
                           else 'warnings')
                    result[tag].append(
                        f'{cam} 时间戳为0(丢帧) {cam_zero[cam]}/{T}')
                if cam_big_off[cam]:
                    tag = ('errors' if cam_big_off[cam] / T > 0.05
                           else 'warnings')
                    result[tag].append(
                        f'{cam} 时间戳偏>100ms: {cam_big_off[cam]}/{T}')

    except Exception as e:
        result['errors'].append(f'h5读取失败: {e}')
        return result

    # ── 3. 视频/txt 帧数对比 (仅看 txt, 不解码视频) ───────────
    for cam in CAMERAS:
        txt_path = ep_dir / 'camera' / cam / f'{cam}.txt'
        if txt_path.exists():
            n = sum(1 for ln in open(txt_path) if ln.strip())
            result['stats'][f'{cam}_txt_frames'] = n
            T = result['stats'].get('h5_frames', 0)
            if T and abs(n - T) > T * 0.1:
                result['warnings'].append(
                    f'{cam} txt帧数({n}) 与 h5帧数({T}) 差异>10%')

    return result


# ────────────────────────────────────────────────────────────────
def main(args):
    data_dir = Path(args.data_dir)
    episodes = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    N = len(episodes)
    print(f"找到 {N} 个 episode，开始检查 (v2)...\n")

    results = []
    for i, ep in enumerate(episodes):
        r = check_episode(ep)
        results.append(r)
        st = '❌' if r['errors'] else ('⚠️ ' if r['warnings'] else '✅')
        s  = r['stats']
        print(f"[{i+1:3d}/{N}] {st} {ep.name}  "
              f"T={s.get('h5_frames','?')} "
              f"miss_a={s.get('miss_action_total',0)}("
              f"h{s.get('miss_action_head',0)}/m{s.get('miss_action_middle',0)}max{s.get('miss_action_middle_max',0)}"
              f"/t{s.get('miss_action_tail',0)})"
              f" usable={s.get('usable_frames','?')}",
              flush=True)

    ok   = [r for r in results if not r['errors'] and not r['warnings']]
    warn = [r for r in results if not r['errors'] and r['warnings']]
    err  = [r for r in results if r['errors']]

    lines = []
    def w(s=''):
        lines.append(s); print(s)

    w('\n' + '=' * 72)
    w('数据集一致性检查报告 V2')
    w('=' * 72)
    w(f'总 episode 数:          {N}')
    w(f'完全正常 (OK):          {len(ok)}')
    w(f'可修复 (WARNING):       {len(warn)}')
    w(f'建议丢弃 (ERROR):       {len(err)}')
    w(f'可用 (OK+WARNING):      {len(ok) + len(warn)} / {N}')

    # 分布
    frames = [r['stats']['h5_frames'] for r in results if 'h5_frames' in r['stats']]
    if frames:
        w(f'\n帧数分布: min={min(frames)} max={max(frames)} '
          f'mean={np.mean(frames):.0f} P50={np.percentile(frames,50):.0f} '
          f'P95={np.percentile(frames,95):.0f}')

    usable = [r['stats']['usable_frames'] for r in results
              if 'usable_frames' in r['stats']]
    if usable:
        w(f'裁剪后可用帧(usable): min={min(usable)} max={max(usable)} '
          f'mean={np.mean(usable):.0f} P50={np.percentile(usable,50):.0f} '
          f'P95={np.percentile(usable,95):.0f}')

    head_all = [r['stats'].get('miss_action_head', 0) for r in results]
    tail_all = [r['stats'].get('miss_action_tail', 0) for r in results]
    mid_all  = [r['stats'].get('miss_action_middle', 0) for r in results]
    mid_mx   = [r['stats'].get('miss_action_middle_max', 0) for r in results]
    if head_all:
        w(f'\naction 首段缺: mean={np.mean(head_all):.1f} max={max(head_all)}')
        w(f'action 末段缺: mean={np.mean(tail_all):.1f} max={max(tail_all)}')
        w(f'action 中段缺: mean={np.mean(mid_all):.1f}  max={max(mid_all)}')
        w(f'action 中段最长连续: mean={np.mean(mid_mx):.1f}  max={max(mid_mx)}')

    # ERROR 详情
    if err:
        w('\nERROR episodes:')
        for r in err[:30]:
            w(f'  ❌ {r["name"]}')
            for e in r['errors']:
                w(f'      {e}')
        if len(err) > 30:
            w(f'  ... 共 {len(err)} 条')

    # 输出
    json_out = Path(args.output_report).with_suffix('.json')
    with open(json_out, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    w(f'\n详细 JSON: {json_out}')

    drop_list = Path(args.output_report).parent / 'episodes_to_drop_v2.txt'
    with open(drop_list, 'w') as f:
        for r in err:
            f.write(r['name'] + '\n')
    w(f'丢弃列表:   {drop_list}')

    # 输出"可修复"列表（用于 convert 脚本的 trim+fill 策略）
    fixable_list = Path(args.output_report).parent / 'episodes_fixable_v2.json'
    fixable = []
    for r in (ok + warn):
        s = r['stats']
        fixable.append({
            'name': r['name'],
            'path': r['path'],
            'h5_frames': s.get('h5_frames'),
            'trim_start': s.get('miss_action_head', 0),
            'trim_end':   s.get('miss_action_tail', 0),
            'usable_frames': s.get('usable_frames'),
            'middle_miss_max': s.get('miss_action_middle_max', 0),
        })
    with open(fixable_list, 'w') as f:
        json.dump(fixable, f, indent=2, default=str)
    w(f'可修复清单: {fixable_list}')

    with open(args.output_report, 'w') as f:
        f.write('\n'.join(lines))
    w(f'报告: {args.output_report}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_report',
        default='./dataset_consistency_report_v2.txt')
    main(parser.parse_args())
