#!/usr/bin/env bash
# ==============================================================================
# 4-组 ACT 对照训练脚本（单机 8 卡，占用 GPU 0-3，并行跑 4 个独立实验）
#
# 实验矩阵（本地 8×H20-3e，GPU 0-3 已被 openpi 占用 131GB/卡，故用 GPU 4-7）：
#   ┌────────┬─────────────┬────────────┬────────────────────────────────┐
#   │  GPU   │  kl_weight  │ chunk_size │            备注                │
#   ├────────┼─────────────┼────────────┼────────────────────────────────┤
#   │   4    │     10      │     20     │ baseline（已训过的配置）       │
#   │   5    │      1      │     20     │ 降 KL 权重，解决 latent 坍缩   │
#   │   6    │     10      │     30     │ 延长 chunk，看是否帮助长程     │
#   │   7    │      1      │     30     │ 组合改动 (GPU7 已占 7GB 够用)  │
#   └────────┴─────────────┴────────────┴────────────────────────────────┘
#
# 运行方式（本地即训练机）：
#   bash run_ablation_4gpu.sh
#
# 观察日志：
#   tail -f logs_ablation/kl10_ch20.log
#   nvidia-smi
#
# 停止所有：
#   pkill -f 'imitate_episodes.py.*sorting_blocks_abl'
# ==============================================================================

set -u  # 未定义变量报错；不用 -e，避免一个后台任务失败就终止脚本

# ------------------------------------------------------------------ 基本配置
REPO_DIR="/home/liumouyun/Downloads/ACT-plus-plus"                     # 本地仓库路径
CKPT_ROOT="/home/liumouyun/extended_storage/liumouyun/checkpoints"     # ckpt 根目录（与已有训练一致）
LOG_DIR="${REPO_DIR}/logs_ablation"                                    # 日志目录
CONDA_ENV="aloha"                                                      # 训练 conda 环境名
TASK_NAME="sorting_blocks"                            # constants.py 中的任务名
POLICY_CLASS="ACT"
BATCH_SIZE=8
LR=1e-5
SEED=0
NUM_STEPS=500000
VALIDATE_EVERY=1000
SAVE_EVERY=5000
EVAL_EVERY=500001            # 训练中不做在线 eval（我们本地离线比较）
HIDDEN_DIM=512
DIM_FF=3200

mkdir -p "${LOG_DIR}"
mkdir -p "${CKPT_ROOT}"

# ------------------------------------------------------------ 激活 conda 环境
# 直接 source 避免 shell 不是交互式时 conda activate 失败
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

cd "${REPO_DIR}"

# ---------------------------------------------------------- 启动单次训练函数
# 用法：launch_one <gpu_id> <kl_weight> <chunk_size> <tag>
launch_one() {
    local gpu=$1
    local kl=$2
    local chunk=$3
    local tag=$4

    local ckpt_dir="${CKPT_ROOT}/sorting_blocks_abl_${tag}"
    local log_file="${LOG_DIR}/${tag}.log"

    mkdir -p "${ckpt_dir}"

    echo "[launch] GPU=${gpu}  tag=${tag}  kl=${kl}  chunk=${chunk}"
    echo "         ckpt_dir = ${ckpt_dir}"
    echo "         log      = ${log_file}"

    CUDA_VISIBLE_DEVICES=${gpu} nohup python3 imitate_episodes.py \
        --task_name       "${TASK_NAME}" \
        --ckpt_dir        "${ckpt_dir}" \
        --policy_class    "${POLICY_CLASS}" \
        --kl_weight       "${kl}" \
        --chunk_size      "${chunk}" \
        --hidden_dim      "${HIDDEN_DIM}" \
        --dim_feedforward "${DIM_FF}" \
        --batch_size      "${BATCH_SIZE}" \
        --lr              "${LR}" \
        --seed            "${SEED}" \
        --num_steps       "${NUM_STEPS}" \
        --eval_every      "${EVAL_EVERY}" \
        --validate_every  "${VALIDATE_EVERY}" \
        --save_every      "${SAVE_EVERY}" \
        > "${log_file}" 2>&1 &

    local pid=$!
    echo "         pid      = ${pid}"
    echo "${pid}" > "${ckpt_dir}/train.pid"
    # 稍微错开启动时间，避免 4 个进程同时 import / 读 dataset 造成 IO 抖动
    sleep 5
}

# --------------------------------------------------------------- 启动 4 组
# GPU 0-3 被 openpi 占满，GPU 4-6 全空，GPU 7 还剩 ~136GB，均够用
launch_one 4 10 20 "kl10_ch20"
launch_one 5  1 20 "kl1_ch20"
launch_one 6 10 30 "kl10_ch30"
launch_one 7  1 30 "kl1_ch30"

echo ""
echo "================================================================"
echo "4 组训练已全部后台启动。"
echo "日志目录: ${LOG_DIR}"
echo ""
echo "快速查看进度："
echo "  tail -f ${LOG_DIR}/kl10_ch20.log"
echo "  watch -n 5 'grep -E \"val_loss|val/l1|step\" ${LOG_DIR}/*.log | tail -40'"
echo "  nvidia-smi"
echo ""
echo "停止全部："
echo "  pkill -f 'imitate_episodes.py.*sorting_blocks_abl'"
echo "================================================================"
