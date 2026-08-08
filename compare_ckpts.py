#!/usr/bin/env python3
"""
离线对比两个 ACT checkpoint 的表现差异。

功能:
  1. 从验证数据集采样 N 条 episode
  2. 两个模型分别在每个时间步做推理 (预测 chunk_size 步动作)
  3. 统计:
     - 每关节 L1 误差 (action vs GT)
     - chunk MSE (整段预测 vs GT 未来 chunk_size 步)
     - gripper open/close 事件时间偏差
     - jerk (平滑度) 指标
  4. 保存对比表 + 前 K 条 episode 的轨迹叠加图

用法示例:
  python compare_ckpts.py \
      --ckpt_a /home/liumouyun/extended_storage/liumouyun/checkpoints/sorting_blocks2 \
      --ckpt_b /home/liumouyun/extended_storage/liumouyun/checkpoints/sorting_blocks_merged \
      --name_a  "blocks_294"  --name_b "blocks_merged_644" \
      --dataset_dirs \
        /home/liumouyun/extended_storage/liumouyun/Datas/record_sorting_blocks_20260416/act_dataset \
      --num_episodes 10 --stride 10 \
      --out_dir ./ckpt_compare_result
"""
import os, sys, argparse, pickle, json, glob
import numpy as np
import h5py
import cv2
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from policy import ACTPolicy

IMG_SIZE = (480, 640)   # (H, W) 与部署端一致
GRIPPER_OPEN_THR = -0.3  # < 即视作"关闭"


# ────────────────────────────────────────────────────────────────
# 载入 & 预处理
# ────────────────────────────────────────────────────────────────
def load_policy(ckpt_dir, ckpt_name, device):
    with open(os.path.join(ckpt_dir, 'config.pkl'), 'rb') as f:
        cfg = pickle.load(f)
    policy_cfg = cfg['policy_config']
    policy = ACTPolicy(policy_cfg)
    sd = torch.load(os.path.join(ckpt_dir, ckpt_name), map_location=device)
    policy.deserialize(sd)
    policy.to(device).eval()
    with open(os.path.join(ckpt_dir, 'dataset_stats.pkl'), 'rb') as f:
        stats = pickle.load(f)
    stats_t = {k: torch.from_numpy(v).float().to(device) for k, v in stats.items()}
    return policy, policy_cfg, stats_t


def list_episodes(dataset_dirs):
    files = []
    for d in dataset_dirs:
        files.extend(sorted(glob.glob(os.path.join(d, 'episode_*.hdf5'))))
    return files


def load_episode(path, camera_names):
    """返回 images[name]=(T,H,W,3) uint8, qpos=(T,D), action=(T,D)"""
    with h5py.File(path, 'r') as f:
        qpos = f['observations/qpos'][()].astype(np.float32)
        action = f['action'][()].astype(np.float32)
        imgs = {}
        for n in camera_names:
            imgs[n] = f[f'observations/images/{n}'][()]
    return imgs, qpos, action


def preprocess_frame(images_at_t, camera_names, device):
    """images_at_t[name]=(H,W,3) uint8 → tensor (1, n_cam, 3, 480, 640) float"""
    resized = []
    for n in camera_names:
        img = images_at_t[n]
        if img.shape[:2] != IMG_SIZE:
            img = cv2.resize(img, (IMG_SIZE[1], IMG_SIZE[0]))
        img = np.transpose(img, (2, 0, 1))
        resized.append(img)
    arr = np.stack(resized).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).to(device)


@torch.inference_mode()
def infer_chunk(policy, stats, qpos_np, img_tensor):
    qpos_t = torch.from_numpy(qpos_np).float().unsqueeze(0).to(img_tensor.device)
    qpos_t = (qpos_t - stats['qpos_mean']) / stats['qpos_std']
    a_hat = policy(qpos_t, img_tensor)  # (1, T, D)
    a_hat = a_hat * stats['action_std'] + stats['action_mean']
    return a_hat[0].cpu().numpy()


# ────────────────────────────────────────────────────────────────
# 指标
# ────────────────────────────────────────────────────────────────
def chunk_errors(pred, gt):
    """pred, gt: (T_pred, D). T 取两者最小. 返回 per-step per-joint |err|"""
    T = min(len(pred), len(gt))
    return np.abs(pred[:T] - gt[:T])   # (T, D)


def gripper_events(seq, thr=GRIPPER_OPEN_THR):
    """夹爪关闭事件的时间索引 (穿过阈值，从开→关)"""
    closed = seq < thr
    events = []
    for t in range(1, len(closed)):
        if closed[t] and not closed[t - 1]:
            events.append(t)
    return events


def match_events(gt_events, pred_events, tol=5):
    """返回匹配数、GT 总数、pred 总数、匹配对的时间差"""
    gt_used = [False] * len(gt_events)
    matched = 0
    diffs = []
    for pe in pred_events:
        best, best_j = tol + 1, -1
        for j, ge in enumerate(gt_events):
            if gt_used[j]:
                continue
            d = abs(pe - ge)
            if d < best:
                best, best_j = d, j
        if best_j >= 0 and best <= tol:
            gt_used[best_j] = True
            matched += 1
            diffs.append(pe - gt_events[best_j])
    return matched, len(gt_events), len(pred_events), diffs


def jerk(action_seq):
    """近似 jerk = 三阶差分平均绝对值 per-joint"""
    if len(action_seq) < 4:
        return np.zeros(action_seq.shape[1])
    d3 = np.diff(action_seq, n=3, axis=0)
    return np.mean(np.abs(d3), axis=0)


# ────────────────────────────────────────────────────────────────
# 评估单个 ckpt
# ────────────────────────────────────────────────────────────────
def evaluate_ckpt(name, ckpt_dir, ckpt_name, episodes, stride, camera_names,
                  max_frames, plot_episodes, out_dir, device):
    print(f"\n===== 评估 {name}  ({ckpt_dir}) =====")
    policy, cfg, stats = load_policy(ckpt_dir, ckpt_name, device)
    chunk_size = cfg['num_queries']
    print(f"  chunk_size={chunk_size}")

    per_ep_metrics = []
    # 用于画图: {ep_idx: {'gt':..., 'pred_chunk0':..., 'ts':...}}
    traj_cache = {}

    for ep_i, ep_path in enumerate(episodes):
        try:
            imgs, qpos, action = load_episode(ep_path, camera_names)
        except Exception as e:
            print(f"  ⚠️ 跳过 {ep_path}: {e}")
            continue
        T = len(qpos)
        # 采样时间步
        ts = list(range(0, min(T - 1, max_frames), stride))
        per_joint_l1 = []         # 预测的 chunk[0] 单步误差
        per_chunk_mse = []        # 整段 chunk 误差
        first_action_pred = []    # 每个 t 的 chunk[0]，用于画图 & jerk
        first_ts = []

        for t in ts:
            img_t = preprocess_frame({n: imgs[n][t] for n in camera_names},
                                     camera_names, device)
            pred_chunk = infer_chunk(policy, stats, qpos[t], img_t)  # (chunk_size, D)
            gt_chunk = action[t:t + chunk_size]
            err = chunk_errors(pred_chunk, gt_chunk)
            per_joint_l1.append(err[0])           # 单步(chunk[0])误差
            per_chunk_mse.append(np.mean(err ** 2, axis=0))
            first_action_pred.append(pred_chunk[0])
            first_ts.append(t)

        per_joint_l1 = np.stack(per_joint_l1)     # (N, D)
        per_chunk_mse = np.stack(per_chunk_mse)   # (N, D)
        first_action_pred = np.stack(first_action_pred)  # (N, D)

        # gripper 事件 (用 chunk[0] 重建的稀疏预测序列，与 GT 对应时间戳)
        gt_action_sampled = action[first_ts]
        gl_events_gt = gripper_events(gt_action_sampled[:, 14])
        gl_events_pr = gripper_events(first_action_pred[:, 14])
        gr_events_gt = gripper_events(gt_action_sampled[:, 15])
        gr_events_pr = gripper_events(first_action_pred[:, 15])

        per_ep_metrics.append({
            'ep_path': ep_path,
            'T': T,
            'n_frames_eval': len(ts),
            'per_joint_l1': per_joint_l1,
            'per_chunk_mse': per_chunk_mse,
            'first_action_pred': first_action_pred,
            'first_ts': np.array(first_ts),
            'gt_action_sampled': gt_action_sampled,
            'gl_events_gt': gl_events_gt,
            'gl_events_pr': gl_events_pr,
            'gr_events_gt': gr_events_gt,
            'gr_events_pr': gr_events_pr,
            'jerk_pred': jerk(first_action_pred),
            'jerk_gt': jerk(gt_action_sampled),
        })

        if ep_i in plot_episodes:
            traj_cache[ep_i] = per_ep_metrics[-1]

        print(f"  [ep {ep_i:2d}] T={T} "
              f"l1_mean={per_joint_l1.mean():.4f} "
              f"arm_l1={per_joint_l1[:,:14].mean():.4f} "
              f"grip_l1={per_joint_l1[:,14:].mean():.4f}")

    del policy
    torch.cuda.empty_cache()

    # 聚合
    all_l1 = np.concatenate([m['per_joint_l1'] for m in per_ep_metrics])   # (sumN, D)
    all_mse = np.concatenate([m['per_chunk_mse'] for m in per_ep_metrics]) # (sumN, D)

    gl_total = {'matched': 0, 'gt': 0, 'pr': 0, 'diffs': []}
    gr_total = {'matched': 0, 'gt': 0, 'pr': 0, 'diffs': []}
    for m in per_ep_metrics:
        mm, g, p, d = match_events(m['gl_events_gt'], m['gl_events_pr'])
        gl_total['matched'] += mm; gl_total['gt'] += g; gl_total['pr'] += p
        gl_total['diffs'].extend(d)
        mm, g, p, d = match_events(m['gr_events_gt'], m['gr_events_pr'])
        gr_total['matched'] += mm; gr_total['gt'] += g; gr_total['pr'] += p
        gr_total['diffs'].extend(d)

    summary = {
        'name': name,
        'ckpt_dir': ckpt_dir,
        'n_episodes': len(per_ep_metrics),
        'n_frames_total': int(all_l1.shape[0]),
        'per_joint_l1_mean': all_l1.mean(axis=0).tolist(),
        'per_joint_l1_p95':  np.percentile(all_l1, 95, axis=0).tolist(),
        'per_joint_mse_mean': all_mse.mean(axis=0).tolist(),
        'overall_l1_mean': float(all_l1.mean()),
        'overall_l1_arm':  float(all_l1[:, :14].mean()),
        'overall_l1_grip': float(all_l1[:, 14:].mean()),
        'overall_mse_mean': float(all_mse.mean()),
        'gripper_L_match': gl_total, 'gripper_R_match': gr_total,
        'jerk_pred_mean': float(np.mean([m['jerk_pred'].mean() for m in per_ep_metrics])),
        'jerk_gt_mean':   float(np.mean([m['jerk_gt'].mean()   for m in per_ep_metrics])),
    }
    # 保存
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f'metrics_{name}.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    return summary, traj_cache


# ────────────────────────────────────────────────────────────────
# 轨迹对比图
# ────────────────────────────────────────────────────────────────
def plot_compare(name_a, name_b, traj_a, traj_b, out_dir):
    ep_ids = sorted(set(traj_a.keys()) & set(traj_b.keys()))
    for ep in ep_ids:
        ma, mb = traj_a[ep], traj_b[ep]
        ts = ma['first_ts']
        gt = ma['gt_action_sampled']  # 两者 gt 相同
        pred_a = ma['first_action_pred']
        pred_b = mb['first_action_pred']
        D = gt.shape[1]

        fig, axes = plt.subplots(4, 4, figsize=(18, 11), sharex=True)
        axes = axes.flatten()
        joint_labels = [f'arm_l_{i+1}' for i in range(7)] + \
                       [f'arm_r_{i+1}' for i in range(7)] + \
                       ['grip_L', 'grip_R']
        for j in range(D):
            ax = axes[j]
            ax.plot(ts, gt[:, j], 'b-',  label='GT', lw=1.5, alpha=0.8)
            ax.plot(ts, pred_a[:, j], 'r--', label=name_a, lw=1.0)
            ax.plot(ts, pred_b[:, j], 'g--', label=name_b, lw=1.0)
            ax.set_title(joint_labels[j] if j < len(joint_labels) else f'j{j}',
                         fontsize=9)
            ax.grid(alpha=0.3)
            if j == 0:
                ax.legend(fontsize=7)
        fig.suptitle(f'Episode #{ep}  —  GT vs {name_a} vs {name_b}', fontsize=12)
        fig.tight_layout()
        fp = os.path.join(out_dir, f'traj_ep{ep}.png')
        fig.savefig(fp, dpi=110)
        plt.close(fig)
        print(f"  🖼 {fp}")


# ────────────────────────────────────────────────────────────────
# 打印对比表
# ────────────────────────────────────────────────────────────────
def print_compare_table(sa, sb):
    print("\n" + "=" * 78)
    print(f"{'指标':<28}{sa['name']:>22}{sb['name']:>22}   Δ")
    print("-" * 78)
    def row(label, va, vb, fmt='{:>22.5f}'):
        d = vb - va
        ds = '  (↓ better)' if d < 0 else '  (↑ worse)'
        print(f"{label:<28}{fmt.format(va):>22}{fmt.format(vb):>22}  {d:+.5f}{ds}")
    row("overall L1 (all)",   sa['overall_l1_mean'], sb['overall_l1_mean'])
    row("overall L1 (arm14)", sa['overall_l1_arm'],  sb['overall_l1_arm'])
    row("overall L1 (grip)",  sa['overall_l1_grip'], sb['overall_l1_grip'])
    row("chunk MSE (all)",    sa['overall_mse_mean'], sb['overall_mse_mean'])
    row("pred jerk mean",     sa['jerk_pred_mean'],   sb['jerk_pred_mean'])
    print(f"{'GT jerk (reference)':<28}{sa['jerk_gt_mean']:>22.5f}")

    print("\n  Gripper L event matching:")
    for s in (sa, sb):
        g = s['gripper_L_match']
        prec = g['matched']/max(g['pr'],1); rec = g['matched']/max(g['gt'],1)
        f1 = 2*prec*rec/max(prec+rec,1e-9)
        md = np.mean(g['diffs']) if g['diffs'] else float('nan')
        print(f"    {s['name']:<20} P={prec:.2f} R={rec:.2f} F1={f1:.2f} "
              f"mean_dt={md:+.1f}步  (matched {g['matched']}/{g['gt']})")

    print("\n  Gripper R event matching:")
    for s in (sa, sb):
        g = s['gripper_R_match']
        prec = g['matched']/max(g['pr'],1); rec = g['matched']/max(g['gt'],1)
        f1 = 2*prec*rec/max(prec+rec,1e-9)
        md = np.mean(g['diffs']) if g['diffs'] else float('nan')
        print(f"    {s['name']:<20} P={prec:.2f} R={rec:.2f} F1={f1:.2f} "
              f"mean_dt={md:+.1f}步  (matched {g['matched']}/{g['gt']})")

    print("\n  Per-joint L1 mean:")
    joint_labels = [f'arm_l_{i+1}' for i in range(7)] + \
                   [f'arm_r_{i+1}' for i in range(7)] + ['grip_L', 'grip_R']
    for j, lab in enumerate(joint_labels):
        va = sa['per_joint_l1_mean'][j]; vb = sb['per_joint_l1_mean'][j]
        d = vb - va
        print(f"    {lab:<10} {va:.4f}   {vb:.4f}   Δ={d:+.4f}")
    print("=" * 78)


# ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt_a', required=True)
    ap.add_argument('--ckpt_b', required=True)
    ap.add_argument('--name_a', default='model_A')
    ap.add_argument('--name_b', default='model_B')
    ap.add_argument('--ckpt_name', default='policy_best.ckpt')
    ap.add_argument('--dataset_dirs', nargs='+', required=True)
    ap.add_argument('--camera_names', nargs='+',
                    default=['hand_left_color', 'hand_right_color', 'head_color'])
    ap.add_argument('--num_episodes', type=int, default=10)
    ap.add_argument('--stride', type=int, default=10,
                    help='每隔多少帧评估一次 (降采样以节省时间)')
    ap.add_argument('--max_frames', type=int, default=200,
                    help='每条 episode 最多评估到第 N 帧')
    ap.add_argument('--plot_k', type=int, default=3,
                    help='保存前 K 条 episode 的轨迹对比图')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out_dir', default='./ckpt_compare_result')
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(args.gpu))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device={device}")

    os.makedirs(args.out_dir, exist_ok=True)

    all_eps = list_episodes(args.dataset_dirs)
    print(f"发现 {len(all_eps)} 条 episode")
    rng = np.random.default_rng(args.seed)
    pick_idx = rng.choice(len(all_eps),
                          size=min(args.num_episodes, len(all_eps)),
                          replace=False)
    pick_idx = sorted(pick_idx.tolist())
    selected = [all_eps[i] for i in pick_idx]
    print("选中:")
    for i, p in zip(pick_idx, selected):
        print(f"  [{i}] {p}")

    plot_episodes = set(range(min(args.plot_k, len(selected))))

    sa, traj_a = evaluate_ckpt(args.name_a, args.ckpt_a, args.ckpt_name,
                               selected, args.stride, args.camera_names,
                               args.max_frames, plot_episodes,
                               args.out_dir, device)
    sb, traj_b = evaluate_ckpt(args.name_b, args.ckpt_b, args.ckpt_name,
                               selected, args.stride, args.camera_names,
                               args.max_frames, plot_episodes,
                               args.out_dir, device)

    print_compare_table(sa, sb)

    plot_compare(args.name_a, args.name_b, traj_a, traj_b, args.out_dir)

    combined = {'A': sa, 'B': sb,
                'selected_episodes': selected,
                'args': vars(args)}
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"\n✅ 结果已写入 {args.out_dir}/")


if __name__ == '__main__':
    main()
