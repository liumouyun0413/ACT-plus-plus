# G2 机器人 ACT 策略全流程文档（积木分拣任务）

> 日期：2026年4月18日  
> 机器人：Agibot G2（双臂，GDK v2.6.3）  
> 任务：sorting_blocks（积木分拣）  

---

## 一、原始数据筛选

### 1.1 数据来源

Agibot G2 机器人遥操作采集，约300条演示（`sorting_blocks_*` 目录），原始采集频率 **30Hz**。

每条演示的目录结构：
```
sorting_blocks_YYYYMMDDHHMMSS/
├── record/aligned_joints.h5     # 关节状态/动作，按帧索引存储（30Hz）
├── camera/hand_left_color/
│   ├── hand_left_color.h265     # 视频流（30fps）
│   └── hand_left_color.txt      # 每帧时间戳
├── camera/hand_right_color/
│   ├── hand_right_color.h265
│   └── hand_right_color.txt
└── camera/head_color/
    ├── head_color.h265
    └── head_color.txt
```

### 1.2 一致性检查

使用 `check_dataset_consistency.py` 对全部数据做自动化检查：

```bash
conda run -n aloha python check_dataset_consistency.py \
  --data_dir /home/liumouyun/extended_storage/liumouyun/Datas/record_sorting_blocks_20260415/record \
  --output_report dataset_consistency_report.txt
```

**检查项目：**
1. 目录完整性（必要文件是否存在）
2. `aligned_joints.h5` 帧数分布（过短/过长异常）
3. 相机视频帧数与 h5 帧数是否匹配
4. 相机时间戳文件行数一致性
5. action 数据异常（全零、NaN、inf、范围超限）
6. qpos 数据异常
7. 夹爪数据异常
8. episode 时长分布统计

### 1.3 剔除的问题数据

共剔除 **5条** 问题数据（详见 `episodes_to_drop.txt`）：

| 序号 | episode 名称 |
|------|-------------|
| 1 | `sorting_blocks_20260415104451` |
| 2 | `sorting_blocks_20260415103834` |
| 3 | `sorting_blocks_20260415135915` |
| 4 | `sorting_blocks_20260415145741` |
| 5 | `sorting_blocks_20260415160055` |

筛选后保留 **294条** 有效演示。

---

## 二、数据格式转换

### 2.1 转换工具与命令

使用 `convert_to_act_hdf5.py` 将原始采集数据转换为 ACT 训练所需的 HDF5 格式。

**关键参数**：`--stride 7`，将 30Hz 降采样到真实控制频率 **~4.3Hz**。

> 频率说明：`aligned_joints.h5` 已将关节数据上采样至 30Hz（与相机同步），但关节控制指令（action）实际频率为 ~4.3Hz（间隔 ~230ms），因此每 ~7 帧 action 才真正变化一次，其余帧为 hold 重复值。直接用 30Hz 数据训练会导致模型学习大量"保持不动"的冗余动作。

```bash
python convert_to_act_hdf5.py \
  --input_dir /home/liumouyun/Downloads/ACT-plus-plus \
  --output_dir /home/liumouyun/extended_storage/liumouyun/Datas/record_sorting_blocks_20260415/act_dataset \
  --cameras hand_left_color hand_right_color head_color \
  --prefix sorting_block \
  --stride 7
```

### 2.2 输出 HDF5 结构（ACT 标准格式）

每条 episode 生成一个 `episode_{i}.hdf5`：

| 键 | 形状 | 类型 | 说明 |
|----|------|------|------|
| `/observations/qpos` | (T, 16) | float32 | 14臂关节 + 2夹爪位置 |
| `/observations/qvel` | (T, 16) | float32 | 关节速度 |
| `/observations/images/hand_left_color` | (T, 480, 640, 3) | uint8 | 左手相机 RGB |
| `/observations/images/hand_right_color` | (T, 480, 640, 3) | uint8 | 右手相机 RGB |
| `/observations/images/head_color` | (T, 480, 640, 3) | uint8 | 头部相机 RGB |
| `/action` | (T, 16) | float32 | 动作指令（14臂 + 2夹爪）|

16维动作向量定义：
- `[0:7]`：左臂 7 关节 (joint1-7)
- `[7:14]`：右臂 7 关节 (joint1-7)
- `[14]`：左夹爪（-0.785=全开, 0.0=全闭）
- `[15]`：右夹爪（-0.785=全开, 0.0=全闭）

### 2.3 数据统计

| 指标 | 值 |
|------|-----|
| 有效 episode 数 | 294 |
| 帧数最小值 | 121 |
| 帧数最大值 | 213 |
| 帧数均值 | 160 |
| 帧数中位数 | 155 |
| **帧数 P95** | **191** |
| 数据频率 | ~4.3 Hz |

---

## 三、数据验证

使用 `visualize_act_dataset.py` 生成可视化视频，验证转换后数据的正确性：

```bash
python visualize_act_dataset.py --episode 0 --output_dir ./vis_videos
```

**可视化内容：**
- **上半部分**：3个相机画面水平拼接（hand_left_color / hand_right_color / head_color）
- **下半部分**：关节角度曲线面板
  - 蓝色线 = qpos（实际位置）
  - 红色线 = action（控制指令）
  - 竖线 = 当前帧
  - 左半 = 左臂7关节 + 左夹爪
  - 右半 = 右臂7关节 + 右夹爪

**验证要点：**
- 图像与关节数据时序对齐
- 夹爪开合动作与画面中抓取动作一致
- action 与 qpos 整体趋势吻合（action 略领先于 qpos）
- 无明显数据异常/跳变

---

## 四、训练

### 4.1 任务配置

在 `constants.py` 中配置：

```python
REAL_TASK_CONFIGS = {
    'sorting_blocks': {
        'dataset_dir': '.../act_dataset',
        'num_episodes': 294,
        'episode_len': 191,        # P95，覆盖95%的演示长度
        'camera_names': ['hand_left_color', 'hand_right_color', 'head_color'],
    },
}
```

### 4.2 训练命令

```bash
cd /home/liumouyun/Downloads/ACT-plus-plus && \
nohup conda run -n aloha env CUDA_VISIBLE_DEVICES=1 python imitate_episodes.py \
  --task_name sorting_blocks \
  --ckpt_dir /home/liumouyun/extended_storage/liumouyun/checkpoints/sorting_blocks \
  --policy_class ACT --kl_weight 10 --chunk_size 20 \
  --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
  --num_steps 200000 --lr 1e-5 --seed 0 \
  --save_every 5000 --validate_every 1000 --eval_every 200001 \
  > /tmp/train_sorting_blocks.log 2>&1 &
```

### 4.3 训练参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `policy_class` | ACT | Action Chunking with Transformers |
| `chunk_size` | 20 | 每次预测20步动作（≈4.7s @4.3Hz）|
| `kl_weight` | 10 | CVAE KL散度权重 |
| `hidden_dim` | 512 | Transformer隐藏维度 |
| `dim_feedforward` | 3200 | FFN维度 |
| `batch_size` | 8 | 批大小 |
| `lr` | 1e-5 | 学习率 |
| `num_steps` | 200000 | 训练总步数 |
| `save_every` | 5000 | 每5000步保存checkpoint |
| `validate_every` | 1000 | 每1000步验证 |
| `eval_every` | 200001 | 不做eval（设为大于总步数）|
| `CUDA_VISIBLE_DEVICES` | 1 | 使用第2张GPU |

### 4.4 模型信息

- **架构**：ACT (Action Chunking with Transformers)
- **参数量**：**106.22M**
- **输入**：3相机图像 (480×640) + 16维关节位置
- **输出**：20步 × 16维动作序列

### 4.5 训练产物

存放于 checkpoint 目录：
- `policy_best.ckpt` — 最佳验证损失模型
- `policy_last.ckpt` — 最后一轮模型
- `config.pkl` — 训练配置（含 policy_config）
- `dataset_stats.pkl` — 数据归一化统计量（qpos_mean/std, action_mean/std）

---

## 五、部署

### 5.1 部署环境

| 项目 | 值 |
|------|-----|
| 机器人 IP | `10.130.96.99` |
| 用户 | `agi` |
| 远程路径 | `/data/agi/liumouyun/act-plus-plus/` |
| Python 环境 | `/data/agi/envs/aloha/bin/python`（Python 3.12）|
| GDK 版本 | v2.6.3 |
| GPU | 机器人端 Jetson/Thor (aarch64) |

### 5.2 一键部署

```bash
bash deploy_to_robot.sh
```

部署内容：推理脚本 + 策略代码 + DETR 模型代码。Checkpoint 手动拷贝到机器人 `./sorting_blocks2/` 目录。

### 5.3 机器人端运行

```bash
# 1. SSH 登录
ssh agi@10.130.96.99

# 2. 环境设置（每次登录）
source ~/app/env.sh
export LD_LIBRARY_PATH=/data/lxl/Isaac-GR00T/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:/usr/local/cuda-13.0/targets/sbsa-linux/lib:/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH

# 3. 复位到初始位姿
python g2_reset_home.py

# 4. 运行推理
python g2_ACT_simple.py \
  --ckpt_dir ./sorting_blocks2 \
  --task_name sorting_blocks \
  --max_chunks 30
```

---

## 六、推理脚本关键设计

推理脚本：`g2_ACT_simple.py`（最佳版本为 `g2_ACT_simple copy.py`）

### 6.1 控制架构

```
获取观测(图像+qpos) → ACT推理(20步chunk) → 100Hz线性插值下发 → 异步预取下一chunk → 无缝衔接
```

### 6.2 核心参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `SERVO_DT` | 0.01s | 100Hz 伺服频率 |
| `interp_steps` | 23 | 每步插值23次，23×10ms=230ms ≈ 1/4.3Hz |
| `ACTION_EMA_ALPHA` | 0.5 | EMA 平滑系数（0=全用历史, 1=全用新值）|
| `MAX_JOINT_DELTA` | 0.5 rad/step | 单步最大关节变化量（安全限制）|
| chunk总步数 | 20×23=460步 | 每chunk执行4.6s |

### 6.3 控制接口

| 控制对象 | API | 说明 |
|----------|-----|------|
| 14臂关节 | `joint_servo_control` | 统一发送，含夹爪关节名（GDK会警告但不影响）|
| 夹爪 | 随 servo 一起发送 | 夹爪关节名 GDK 不识别，会打印警告但不崩溃 |

### 6.4 安全机制

**GDK 真实关节限位**（`safety_clamp` 函数）：

| 关节 | 下限 (rad) | 上限 (rad) |
|------|-----------|-----------|
| joint1, 3, 5 | -2.880 | 2.880 |
| joint2 | -3.142 | 0.087 |
| joint4 | -2.339 | 0.087 |
| joint6 | -1.677 | 2.077 |
| **joint7** | **-1.536** | **1.536** |
| gripper (omnipicker) | -0.785 | 0.0 |

**其他安全措施：**
- `MAX_JOINT_DELTA`：每步关节变化不超过0.5rad
- `try/except`：`joint_servo_control` 失败时跳过该步，不崩溃
- 失败时不更新 `prev_action`，避免下一步突变跳跃

### 6.5 异步预取

- 在当前 chunk 执行到末尾（提前约150ms）时启动后台线程推理下一个 chunk
- 推理耗时约 100ms（含图像获取 ~25ms + 模型推理 ~60ms）
- chunk 间 hold 次数通常为 0，实现无缝衔接

---

## 七、遇到的问题与解决方案

### 7.1 执行速度不匹配

**问题**：训练数据 4.3Hz，初始 `interp_steps=10` 导致执行过快（每chunk仅2s）  
**解决**：`interp_steps` 改为 23，每步230ms匹配训练数据频率

### 7.2 夹爪不开/关不及时

**问题**：复位时夹爪不打开；推理时夹爪关闭滞后  
**解决**：复位脚本中持续发送夹爪指令；推理脚本每步都发送夹爪

### 7.3 joint_servo_control 夹爪警告

**问题**：`Param idx31_gripper_l_inner_joint1 not found`（每步两条警告）  
**原因**：GDK `joint_servo_control` 不识别夹爪关节名  
**影响**：仅打印警告，不影响运行和抓取效果  
**备选方案**：如需消除警告，可将夹爪从 `SERVO_JOINT_NAMES` 移除，改用 `move_ee_pos` 单独发送

### 7.4 关节超限崩溃

**问题**：`idx27_arm_l_joint7 value is out of range` → `RuntimeError: JointServoControl failed`  
**原因**：原 `ARM_JOINT_LIMITS` 为 `[-3.14, 3.14]`，未拦住超出 joint7 真实限位 `[-1.536, 1.536]` 的值  
**解决**：
1. 更新 `ARM_JOINT_LIMITS` 为 GDK 真实限位
2. `send_action` 加 `try/except` 防崩溃
3. 失败时不更新 `prev_action`，防止突变

---

## 八、文件清单

| 文件 | 用途 |
|------|------|
| `check_dataset_consistency.py` | 原始数据一致性检查 |
| `episodes_to_drop.txt` | 需剔除的问题 episode 列表 |
| `convert_to_act_hdf5.py` | 原始数据 → ACT HDF5 格式转换 |
| `visualize_act_dataset.py` | 转换后数据可视化验证 |
| `constants.py` | 任务配置（数据路径、相机名、episode参数）|
| `imitate_episodes.py` | ACT 模型训练 |
| `g2_ACT_simple.py` / `g2_ACT_simple copy.py` | 机器人端推理脚本 |
| `g2_reset_home.py` | 机器人复位到初始位姿 |
| `deploy_to_robot.sh` | 一键部署到机器人 |
| `policy.py` | ACT 策略封装 |
| `detr/` | Transformer 模型代码 |



mkdir -p /home/liumouyun/extended_storage/liumouyun/checkpoints/stack_pickup

# Phase 1
python3 imitate_episodes.py \
    --task_name stack_pickup_cotrain_p1 \
    --policy_class ACT --chunk_size 80 --kl_weight 10 \
    --hidden_dim 512 --dim_feedforward 3200 \
    --batch_size 8 --num_epochs 2000 --lr 1e-5 --seed 0 \
    --ckpt_dir /home/liumouyun/extended_storage/liumouyun/checkpoints/stack_pickup/p1

# Phase 2
python3 imitate_episodes.py \
    --task_name stack_pickup_cotrain_p2 \
    --policy_class ACT --chunk_size 80 --kl_weight 10 \
    --hidden_dim 512 --dim_feedforward 3200 \
    --batch_size 8 --num_epochs 4000 --lr 1e-5 --seed 0 \
    --ckpt_dir         /home/liumouyun/extended_storage/liumouyun/checkpoints/stack_pickup/p2 \
    --resume_ckpt_path /home/liumouyun/extended_storage/liumouyun/checkpoints/stack_pickup/p1/policy_last.ckpt

# Phase 3
python3 imitate_episodes.py \
    --task_name stack_pickup_cotrain_p3 \
    --policy_class ACT --chunk_size 80 --kl_weight 10 \
    --hidden_dim 512 --dim_feedforward 3200 \
    --batch_size 8 --num_epochs 2000 --lr 5e-6 --seed 0 \
    --ckpt_dir         /home/liumouyun/extended_storage/liumouyun/checkpoints/stack_pickup/p3 \
    --resume_ckpt_path /home/liumouyun/extended_storage/liumouyun/checkpoints/stack_pickup/p2/policy_last.ckpt