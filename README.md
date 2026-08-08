# Imitation Learning algorithms and Co-training for Mobile ALOHA


#### Project Website: https://mobile-aloha.github.io/

This repo contains the implementation of ACT, Diffusion Policy and VINN, together with 2 simulated environments:
Transfer Cube and Bimanual Insertion. You can train and evaluate them in sim or real.
For real, you would also need to install [Mobile ALOHA](https://github.com/MarkFzp/mobile-aloha). This repo is forked from the [ACT repo](https://github.com/tonyzhaozh/act).

### Updates:
You can find all scripted/human demo for simulated environments [here](https://drive.google.com/drive/folders/1gPR03v05S1xiInoVJn7G7VJ9pDCnxq9O?usp=share_link).


### Repo Structure
- ``imitate_episodes.py`` Train and Evaluate ACT
- ``policy.py`` An adaptor for ACT policy
- ``detr`` Model definitions of ACT, modified from DETR
- ``sim_env.py`` Mujoco + DM_Control environments with joint space control
- ``ee_sim_env.py`` Mujoco + DM_Control environments with EE space control
- ``scripted_policy.py`` Scripted policies for sim environments
- ``constants.py`` Constants shared across files
- ``utils.py`` Utils such as data loading and helper functions
- ``visualize_episodes.py`` Save videos from a .hdf5 dataset


### Installation

    conda create -n aloha python=3.8.10
    conda activate aloha
    pip install torchvision
    pip install torch
    pip install pyquaternion
    pip install pyyaml
    pip install rospkg
    pip install pexpect
    pip install mujoco==2.3.7
    pip install dm_control==1.0.14
    pip install opencv-python
    pip install matplotlib
    pip install einops
    pip install packaging
    pip install h5py
    pip install ipython
    cd act/detr && pip install -e .

- also need to install https://github.com/ARISE-Initiative/robomimic/tree/r2d2 (note the r2d2 branch) for Diffusion Policy by `pip install -e .`

### Installation on Thor Controller (aarch64 / Miniforge3)

> 适用平台：NVIDIA Thor 控制器，aarch64 (SBSA)，Ubuntu + ROS 2 Jazzy，无外网（内网可访问清华 PyPI 镜像）。  
> 以下步骤已在 `user@ms-thor`（`10.130.97.61`）上完整验证，`import imitate_episodes` 通过。

#### 前置文件（需从开发机传入 Thor，放在同一工作目录，如 `~/Downloads/lmy/`）

| 文件 | 说明 |
|------|------|
| `cu130/torch-2.10.0-cp312-cp312-linux_aarch64.whl` | Thor 专用 PyTorch wheel |
| `cu130/torchvision-0.25.0-cp312-cp312-linux_aarch64.whl` | 对应 torchvision wheel |
| `cu130/nvidia-cudnn-cu12-9.21.0.82-cp312-cp312-linux_aarch64.whl` | cuDNN wheel |
| `cu130/nvidia-cublas-cu12-12.9.1.4-cp312-cp312-linux_aarch64.whl` | cuBLAS wheel |
| `ACT-plus-plus/`（本项目） | 含已修改的 `detr/__init__.py` |
| `robomimic-master/` | 含 `diffusion_policy_nets.py` 的完整分支 |
| `clip-vit-large-patch14.zip` | CLIP 模型（~4GB），用于 robomimic 离线推理 |

传输命令参考（在开发机执行，注意先 `unset LD_LIBRARY_PATH` 再 scp）：

    unset LD_LIBRARY_PATH
    scp -r cu130/ ACT-plus-plus/ robomimic-master/ clip-vit-large-patch14.zip \
        user@<thor-ip>:~/Downloads/lmy/

---

**步骤 0：安装 Miniforge3（首次，若已有则跳过）**

    wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
    bash Miniforge3-Linux-aarch64.sh -b -p ~/miniforge3
    ~/miniforge3/bin/conda init bash && source ~/.bashrc

**步骤 1：创建 conda 环境**

    conda create -n aloha python=3.12 -y
    conda activate aloha

**步骤 2：隔离 ROS PYTHONPATH（防止 ROS 包污染 Python 环境）**

ROS 2 Jazzy 会向 `PYTHONPATH` 注入大量路径，激活 `aloha` 时必须将其屏蔽：

    mkdir -p $CONDA_PREFIX/etc/conda/activate.d $CONDA_PREFIX/etc/conda/deactivate.d

    cat > $CONDA_PREFIX/etc/conda/activate.d/00_isolate_ros.sh << 'EOF'
    export _SAVED_PYTHONPATH="${PYTHONPATH:-}"
    unset PYTHONPATH
    EOF

    cat > $CONDA_PREFIX/etc/conda/deactivate.d/00_restore_ros.sh << 'EOF'
    export PYTHONPATH="${_SAVED_PYTHONPATH:-}"
    unset _SAVED_PYTHONPATH
    EOF

    conda deactivate && conda activate aloha

**步骤 3：升级 pip / 设置清华镜像**

    pip install --upgrade pip wheel setuptools
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

**步骤 4：安装 PyTorch + torchvision（离线 wheel，--no-deps 避免自动拉依赖）**

    cd ~/Downloads/lmy
    pip install --no-deps cu130/torch-2.10.0-cp312-cp312-linux_aarch64.whl
    pip install --no-deps cu130/torchvision-0.25.0-cp312-cp312-linux_aarch64.whl

**步骤 5：安装 CUDA 运行时库（解决 libcudss.so.0 / libcudnn 缺失）**

    # 本地 wheel（优先）
    pip install --no-deps cu130/nvidia-cudnn-cu12-9.21.0.82-cp312-cp312-linux_aarch64.whl
    pip install --no-deps cu130/nvidia-cublas-cu12-12.9.1.4-cp312-cp312-linux_aarch64.whl
    # 补充 cudss / nvrtc（从清华镜像安装）
    pip install nvidia-cudss-cu13 nvidia-cuda-nvrtc-cu12

**步骤 6：将 NVIDIA 库路径写入 conda 激活钩子**

    cat > $CONDA_PREFIX/etc/conda/activate.d/10_nvidia_libs.sh << 'EOF'
    _SP="$CONDA_PREFIX/lib/python3.12/site-packages"
    for _d in "$_SP/torch/lib" $(find "$_SP/nvidia" -maxdepth 2 -name lib -type d 2>/dev/null); do
        [ -d "$_d" ] && export LD_LIBRARY_PATH="${_d}:${LD_LIBRARY_PATH:-}"
    done
    unset _SP _d
    EOF

    conda deactivate && conda activate aloha

**步骤 7：验证 CUDA / cuDNN 可用**

    python -c "
    import torch
    print('torch:', torch.__version__, '  cuda:', torch.version.cuda)
    print('cuda available:', torch.cuda.is_available())
    print('cudnn available:', torch.backends.cudnn.is_available())
    print('device count:', torch.cuda.device_count())
    "
    # 期望: cuda available: True  /  cudnn available: True

**步骤 8：安装 ACT 其余 Python 依赖**

    pip install h5py pyyaml einops matplotlib pyquaternion tqdm \
                rospkg pexpect ipython packaging opencv-python-headless wandb
    pip install mujoco==3.8.0 dm_control

**步骤 9：安装 robomimic（editable，必须使用含 diffusion_policy_nets.py 的版本）**

    cd ~/Downloads/lmy/robomimic-master
    pip install --no-build-isolation -e .
    cd ~/Downloads/lmy

**步骤 10：修复 detr 的绝对 import 问题**

`detr/backbone.py` 等文件使用 `from util.misc import ...` 绝对路径导入，
需在 `detr/__init__.py` 将 detr 目录插入 `sys.path`。
若传入的 `ACT-plus-plus/` 已含此文件则跳过，否则执行：

    cat > ~/Downloads/lmy/ACT-plus-plus/detr/__init__.py << 'EOF'
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    EOF

**步骤 11：本地化 CLIP 模型（robomimic 离线推理）**

    cd ~/Downloads/lmy
    mkdir -p models
    unzip -q clip-vit-large-patch14.zip -d models/

    # 将 robomimic 中的 HuggingFace 路径替换为本地绝对路径
    CLIP_PATH="$HOME/Downloads/lmy/models/clip-vit-large-patch14"
    sed -i "s|\"openai/clip-vit-large-patch14\"|\"${CLIP_PATH}\"|g" \
        robomimic-master/robomimic/utils/lang_utils.py

    # HuggingFace 离线模式（防止运行时联网）
    cat > $CONDA_PREFIX/etc/conda/activate.d/20_hf_offline.sh << 'EOF'
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    EOF

    conda deactivate && conda activate aloha

**步骤 12：验证 imitate_episodes 可导入**

    cd ~/Downloads/lmy/ACT-plus-plus
    python -c "import imitate_episodes; print('OK')"
    # 期望输出: OK

### Example Usages

To set up a new terminal, run:

    conda activate aloha
    cd <path to act repo>

### Simulated experiments (LEGACY table-top ALOHA environments)

We use ``sim_transfer_cube_scripted`` task in the examples below. Another option is ``sim_insertion_scripted``.
To generated 50 episodes of scripted data, run:

    python3 record_sim_episodes.py --task_name sim_transfer_cube_scripted --dataset_dir <data save dir> --num_episodes 50

To can add the flag ``--onscreen_render`` to see real-time rendering.
To visualize the simulated episodes after it is collected, run

    python3 visualize_episodes.py --dataset_dir <data save dir> --episode_idx 0

Note: to visualize data from the mobile-aloha hardware, use the visualize_episodes.py from https://github.com/MarkFzp/mobile-aloha

To train ACT:
    
    # Transfer Cube task
    python3 imitate_episodes.py --task_name sim_transfer_cube_scripted --ckpt_dir <ckpt dir> --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000  --lr 1e-5 --seed 0


To evaluate the policy, run the same command but add ``--eval``. This loads the best validation checkpoint.
The success rate should be around 90% for transfer cube, and around 50% for insertion.
To enable temporal ensembling, add flag ``--temporal_agg``.
Videos will be saved to ``<ckpt_dir>`` for each rollout.
You can also add ``--onscreen_render`` to see real-time rendering during evaluation.

For real-world data where things can be harder to model, train for at least 5000 epochs or 3-4 times the length after the loss has plateaued.
Please refer to [tuning tips](https://docs.google.com/document/d/1FVIZfoALXg_ZkYKaYVh-qOlaXveq5CtvJHXkY25eYhs/edit?usp=sharing) for more info.

### [ACT tuning tips](https://docs.google.com/document/d/1FVIZfoALXg_ZkYKaYVh-qOlaXveq5CtvJHXkY25eYhs/edit?usp=sharing)
TL;DR: if your ACT policy is jerky or pauses in the middle of an episode, just train for longer! Success rate and smoothness can improve way after loss plateaus.
