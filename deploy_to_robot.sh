#!/bin/bash
# 一键部署 ACT 推理代码到机器人控制器
# 用法: bash deploy_to_robot.sh

ROBOT="agi@10.130.96.99"
REMOTE_DIR="/data/agi/liumouyun/act-plus-plus"
LOCAL_DIR="/home/liumouyun/Downloads/ACT-plus-plus"
CKPT_DIR="/home/liumouyun/extended_storage/liumouyun/checkpoints/sorting_blocks"

# 需要先 unset LD_LIBRARY_PATH 避免 SSH OpenSSL 冲突
unset LD_LIBRARY_PATH

echo "📁 创建远程目录..."
ssh $ROBOT "mkdir -p $REMOTE_DIR/detr/models $REMOTE_DIR/detr/util $REMOTE_DIR/checkpoints"

echo "📦 拷贝代码文件..."
scp $LOCAL_DIR/g2_ACT_simple.py $ROBOT:$REMOTE_DIR/
scp $LOCAL_DIR/policy.py $ROBOT:$REMOTE_DIR/
scp $LOCAL_DIR/constants.py $ROBOT:$REMOTE_DIR/
scp $LOCAL_DIR/detr/setup.py $ROBOT:$REMOTE_DIR/detr/
scp $LOCAL_DIR/detr/main.py $ROBOT:$REMOTE_DIR/detr/
scp $LOCAL_DIR/detr/models/__init__.py $ROBOT:$REMOTE_DIR/detr/models/
scp $LOCAL_DIR/detr/models/detr_vae.py $ROBOT:$REMOTE_DIR/detr/models/
scp $LOCAL_DIR/detr/models/backbone.py $ROBOT:$REMOTE_DIR/detr/models/
scp $LOCAL_DIR/detr/models/transformer.py $ROBOT:$REMOTE_DIR/detr/models/
scp $LOCAL_DIR/detr/models/position_encoding.py $ROBOT:$REMOTE_DIR/detr/models/
scp $LOCAL_DIR/detr/models/latent_model.py $ROBOT:$REMOTE_DIR/detr/models/
scp $LOCAL_DIR/detr/util/__init__.py $ROBOT:$REMOTE_DIR/detr/util/
scp $LOCAL_DIR/detr/util/misc.py $ROBOT:$REMOTE_DIR/detr/util/

# checkpoint 已手动拷贝到机器人: $REMOTE_DIR/sorting_blocks2/
# 包含: policy_best.ckpt, policy_last.ckpt, config.pkl, dataset_stats.pkl

echo ""
echo "✅ 部署完成!"
echo ""
echo "在机器人上执行:"
echo "  ssh agi@10.130.96.99"
echo ""
echo "  # 环境设置 (每次登录都需要)"
echo "  source ~/app/env.sh"
echo "  export LD_LIBRARY_PATH=/data/lxl/Isaac-GR00T/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:/usr/local/cuda-13.0/targets/sbsa-linux/lib:/usr/local/cuda-13.0/lib64:\$LD_LIBRARY_PATH"
echo "  export PYTHON=/data/agi/envs/aloha/bin/python"
echo ""
echo "  # 首次: 安装 detr 包"
echo "  cd $REMOTE_DIR/detr && \$PYTHON -m pip install -e . && cd .."
echo ""
echo "  # 运行推理"
echo "  cd $REMOTE_DIR"
echo "  \$PYTHON g2_ACT_simple.py --ckpt_dir ./sorting_blocks2 --task_name sorting_blocks --max_chunks 10"
