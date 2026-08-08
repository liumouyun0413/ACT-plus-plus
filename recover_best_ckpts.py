#!/usr/bin/env python3
"""
从 wandb 拉每个 ablation run 的 val_loss 曲线，找出 val_loss 最低的 step，
然后把对应的 policy_step_{best}_seed_0.ckpt 复制成 policy_best.ckpt。

为何需要：
  ablation 训练在 step=500000 的 final eval_bc() 阶段因 ModuleNotFoundError
  ('aloha_scripts') 崩溃，main() 中保存 policy_best.ckpt 的代码没执行到，
  所以需要从 wandb 历史里挑出最佳 step。

用法：
  python recover_best_ckpts.py            # 只打印不复制（dry-run）
  python recover_best_ckpts.py --apply    # 实际执行复制
  python recover_best_ckpts.py --apply --link    # 用硬链接代替复制（省 1.6GB）

依赖：pip install wandb
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

# 仓库目录下有 ./wandb/ 训练日志目录，会遮蔽真正的 wandb 包；先把当前目录从
# sys.path 里去掉，确保 import 到的是 site-packages 里的 wandb。
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]

CKPT_ROOT = Path("/home/liumouyun/extended_storage/liumouyun/checkpoints")
WANDB_ENTITY = "liumouyun0413-usst"
WANDB_PROJECT = "G2_aloha"

# run_name -> ckpt 子目录名（这里恰好同名）
RUNS = [
    "sorting_blocks_abl_kl10_ch20",
    "sorting_blocks_abl_kl1_ch20",
    "sorting_blocks_abl_kl10_ch30",
    "sorting_blocks_abl_kl1_ch30",
]

# val_loss 在 wandb 里被重命名为 'val_loss'（imitate_episodes.py L582）
VAL_LOSS_KEY = "val_loss"
SAVE_EVERY = 5000  # 与 run_ablation_4gpu.sh 中一致；ckpt 仅在 step % save_every==0 存


def find_best_step(run, save_every: int = SAVE_EVERY):
    """从 run 的 history 中找 val_loss 最低且 step 是 save_every 整数倍的点。"""
    # 不用 pandas（环境里没装）；返回的是 list[dict]
    rows = run.history(keys=["_step", VAL_LOSS_KEY], pandas=False, samples=100000)
    if not rows:
        return None, None

    points = []  # (step, val_loss)
    for r in rows:
        v = r.get(VAL_LOSS_KEY)
        s = r.get("_step")
        if v is None or s is None:
            continue
        try:
            v = float(v)
            s = int(s)
        except (TypeError, ValueError):
            continue
        points.append((s, v))

    if not points:
        return None, None

    saved_points = [(s, v) for s, v in points if s % save_every == 0]
    if not saved_points:
        # 兜底：取全局最小，再取最近的 ckpt step
        s, v = min(points, key=lambda x: x[1])
        return (s // save_every) * save_every, v

    s, v = min(saved_points, key=lambda x: x[1])
    return s, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="实际复制（默认 dry-run）")
    ap.add_argument("--link", action="store_true", help="用硬链接代替复制")
    ap.add_argument("--overwrite", action="store_true", help="覆盖已有的 policy_best.ckpt")
    args = ap.parse_args()

    try:
        import wandb
    except ImportError:
        print("[err] pip install wandb")
        sys.exit(1)

    api = wandb.Api()
    print(f"[wandb] querying {WANDB_ENTITY}/{WANDB_PROJECT} ...\n")

    # 一次拉所有 run，按 name 索引（避免每个 run_name 都查一次）
    all_runs = list(api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}"))
    runs_by_name = {}
    for r in all_runs:
        # 同名可能多次，取最新的一次
        if r.name not in runs_by_name or r.created_at > runs_by_name[r.name].created_at:
            runs_by_name[r.name] = r

    summary = []
    for run_name in RUNS:
        ckpt_dir = CKPT_ROOT / run_name
        print(f"=== {run_name} ===")
        if not ckpt_dir.is_dir():
            print(f"  [skip] ckpt_dir 不存在: {ckpt_dir}")
            summary.append((run_name, None, None, "missing-dir"))
            continue

        run = runs_by_name.get(run_name)
        if run is None:
            print(f"  [skip] wandb 里找不到 run: {run_name}")
            summary.append((run_name, None, None, "missing-run"))
            continue
        print(f"  wandb run: {run.id}  state={run.state}")

        best_step, best_val = find_best_step(run)
        if best_step is None:
            print("  [skip] 没拉到 val_loss 数据")
            summary.append((run_name, None, None, "no-val"))
            continue
        print(f"  best  step={best_step}  val_loss={best_val:.5f}")

        src = ckpt_dir / f"policy_step_{best_step}_seed_0.ckpt"
        dst = ckpt_dir / "policy_best.ckpt"

        if not src.exists():
            print(f"  [skip] 找不到对应 ckpt: {src.name}")
            summary.append((run_name, best_step, best_val, "missing-ckpt"))
            continue

        if dst.exists() and not args.overwrite:
            print(f"  [skip] {dst.name} 已存在（用 --overwrite 覆盖）")
            summary.append((run_name, best_step, best_val, "exists"))
            continue

        action = "link" if args.link else "copy"
        print(f"  -> {action}  {src.name}  ->  {dst.name}")

        if args.apply:
            if dst.exists():
                dst.unlink()
            if args.link:
                os.link(src, dst)
            else:
                shutil.copy2(src, dst)
            print(f"  [ok] {action} done")
            summary.append((run_name, best_step, best_val, action))
        else:
            summary.append((run_name, best_step, best_val, "dry-run"))
        print()

    print("\n================ Summary ================")
    print(f"{'run':<35} {'best_step':>10} {'val_loss':>12}  status")
    for name, step, val, status in summary:
        s_step = str(step) if step is not None else "-"
        s_val = f"{val:.5f}" if val is not None else "-"
        print(f"{name:<35} {s_step:>10} {s_val:>12}  {status}")

    if not args.apply:
        print("\n(dry-run) 加 --apply 实际执行复制；可加 --link 用硬链接省空间。")


if __name__ == "__main__":
    main()
