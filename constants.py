import pathlib
import os

### Task parameters
DATA_DIR = '/home/zfu/interbotix_ws/src/act/data' if os.getlogin() == 'zfu' else '/scr/tonyzhao/datasets'

# ---- 扩展盘根目录（所有大文件统一放这里）----
EXT_ROOT = '/home/liumouyun/extended_storage/liumouyun'
DATA_ROOT = f'{EXT_ROOT}/Datas'
CKPT_ROOT = f'{EXT_ROOT}/checkpoints'

# stack_pickup Phase 1 三阶段数据目录（Phase 1 精简方案：无 Stage-D 故意失败）
# Stage-A 单物 150 条；Stage-B 多物无堆叠 30 条；Stage-C 两层堆叠 30 条；合计 210 条
# 自然失败恢复（`_recov` 后缀）混在 A/B/C 三个目录内，不单独成 Stage-D
_STACK_DATASET_DIRS = [
    f'{DATA_ROOT}/stack_pickup/stage_A_single/act_dataset',
    f'{DATA_ROOT}/stack_pickup/stage_B_multi_flat/act_dataset',
    f'{DATA_ROOT}/stack_pickup/stage_C_two_stack/act_dataset',
]
_STACK_STATS_DIRS = [f'{DATA_ROOT}/stack_pickup/stage_C_two_stack/act_dataset']
_STACK_CAMERAS = ['hand_left_color', 'hand_right_color', 'head_color']

SIM_TASK_CONFIGS = {
    'sim_transfer_cube_scripted':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },

    'sim_transfer_cube_human':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_human',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top']
    },

    'sim_insertion_scripted': {
        'dataset_dir': DATA_DIR + '/sim_insertion_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },

    'sim_insertion_human': {
        'dataset_dir': DATA_DIR + '/sim_insertion_human',
        'num_episodes': 50,
        'episode_len': 500,
        'camera_names': ['top']
    },
    'all': {
        'dataset_dir': DATA_DIR + '/',
        'num_episodes': None,
        'episode_len': None,
        'name_filter': lambda n: 'sim' not in n,
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    },

    'sim_transfer_cube_scripted_mirror':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted_mirror',
        'num_episodes': None,
        'episode_len': 400,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },

    'sim_insertion_scripted_mirror': {
        'dataset_dir': DATA_DIR + '/sim_insertion_scripted_mirror',
        'num_episodes': None,
        'episode_len': 400,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },

}

### Real robot task configs
REAL_TASK_CONFIGS = {
    # 积木分拣任务，合并两批数据: 0415(294条) + 0416(350条) = 644条
    # 数据频率: ~4.3 Hz (stride=7 从30Hz降采样)
    # 0415帧数: min=121, max=213, mean=160, P95=191
    # 0416帧数: min=127, max=174, mean=142, P95=155
    # episode_len 取两批P95较大值=191
    # 建议 chunk_size=20（覆盖约4.7s动作）
    'sorting_blocks': {
        'dataset_dir': [
            '/home/liumouyun/extended_storage/liumouyun/Datas/record_sorting_blocks_20260415/act_dataset',
            '/home/liumouyun/extended_storage/liumouyun/Datas/record_sorting_blocks_20260416/act_dataset',
        ],
        'num_episodes': 644,
        'episode_len': 191,
        'camera_names': ['hand_left_color', 'hand_right_color', 'head_color'],
    },

    # ------------------------------------------------------------------
    # 堆叠抓取（泡沫 / 塑料袋 / 物料板）多源 co-training 配置 - Phase 1
    # ------------------------------------------------------------------
    # Phase 1 精简策略（详见 数据采集SOP_Phase1_单件.md）：
    #   - 不采双臂协作剧本（留给 Phase 2 Stage-C'）
    #   - 不采故意失败/干扰（改为"自然失败保留" _recov 后缀，混入 A/B/C）
    #   - 不采三层堆叠（留给 Phase 2）
    #
    # 三个数据源的角色:
    #   Stage-A: 桌面单件随机姿态（各物体轮换）      150 条  ← 基础技能（左75/右75）
    #   Stage-B: 多件散落无堆叠（2~3 件）             30 条  ← 多物首抓选择
    #   Stage-C: 两层堆叠（3 种组合各 10）            30 条  ← 堆叠目标任务
    #                                              -----
    #                                  合计          210 条
    #
    # sample_weights 顺序严格对齐 dataset_dir: [A, B, C]
    #
    # stats_dir 仅指向 Stage-C（目标任务），避免辅助数据统计污染 normalize
    #
    # ⚠ 前置条件:
    #   1. 三个目录下的 hdf5 必须共享相同的 camera_names / action_dim / qpos_dim
    #   2. episode_len 取三源 P95 的较大值，chunk_size 建议 60~100 并开 --temporal_agg
    #   3. Phase 1 验收通过后进 Phase 2（多件场景），用 p3/policy_best.ckpt warm-start
    # ------------------------------------------------------------------
    'stack_pickup_cotrain': {
        # 3 个数据源顺序: [A 单件, B 多件无堆叠, C 两层堆叠]
        'dataset_dir': _STACK_DATASET_DIRS,
        # 主配方 [A, B, C] = [2, 2, 4]（偏 C 主任务）
        'sample_weights': [2, 2, 4],
        'stats_dir': _STACK_STATS_DIRS,
        'num_episodes': 210,
        'episode_len':  600,
        'camera_names': _STACK_CAMERAS,
    },

    # ------------------------------------------------------------------
    # 课程式 warm-start 三段训练（Phase 1 内部的 p1/p2/p3）
    # ------------------------------------------------------------------
    # 三个 task 共享同一批 dataset_dir / stats_dir / camera_names，
    # 仅 sample_weights 不同，用于在不同训练阶段切换数据配方。
    #
    # 训练接力方式（总计 ~18000 epoch）:
    #   p1 (0     ~  8000): --task_name stack_pickup_cotrain_p1  从零训
    #   p2 (8000  ~ 14000): --task_name stack_pickup_cotrain_p2  --resume_ckpt_path .../p1/policy_best.ckpt
    #   p3 (14000 ~ 18000): --task_name stack_pickup_cotrain_p3  --resume_ckpt_path .../p2/policy_best.ckpt
    #
    # 权重含义 [A, B, C]（Phase 1 无 Stage-D 维度）:
    #   p1 [3,2,2]  A 主导    → 打基础，先学单件抓取原子技能
    #   p2 [2,2,4]  C 主导    → 收敛目标任务（两层堆叠）
    #   p3 [1,2,4]  保持偏 C  → 继续收敛，小 lr 抛光
    # ------------------------------------------------------------------
    'stack_pickup_cotrain_p1': {
        # p1: A 主导，打基础
        'dataset_dir':   _STACK_DATASET_DIRS,
        'sample_weights': [3, 2, 2],
        'stats_dir':     _STACK_STATS_DIRS,
        'num_episodes': 210,
        'episode_len':  600,
        'camera_names': _STACK_CAMERAS,
    },

    'stack_pickup_cotrain_p2': {
        # p2: C 主导，主配方（warm-start 自 p1）
        'dataset_dir':   _STACK_DATASET_DIRS,
        'sample_weights': [2, 2, 4],
        'stats_dir':     _STACK_STATS_DIRS,
        'num_episodes': 210,
        'episode_len':  600,
        'camera_names': _STACK_CAMERAS,
    },

    'stack_pickup_cotrain_p3': {
        # p3: 保持偏 C，小 lr 抛光（warm-start 自 p2）
        'dataset_dir':   _STACK_DATASET_DIRS,
        'sample_weights': [1, 2, 4],
        'stats_dir':     _STACK_STATS_DIRS,
        'num_episodes': 210,
        'episode_len':  600,
        'camera_names': _STACK_CAMERAS,
    },
}

### Simulation envs fixed constants
DT = 0.02
FPS = 50
JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239,  0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]

XML_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/assets/' # note: absolute path

# Left finger position limits (qpos[7]), right_finger = -1 * left_finger
MASTER_GRIPPER_POSITION_OPEN = 0.02417
MASTER_GRIPPER_POSITION_CLOSE = 0.01244
PUPPET_GRIPPER_POSITION_OPEN = 0.05800
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

# Gripper joint limits (qpos[6])
MASTER_GRIPPER_JOINT_OPEN = -0.8
MASTER_GRIPPER_JOINT_CLOSE = -1.65
PUPPET_GRIPPER_JOINT_OPEN = 1.4910
PUPPET_GRIPPER_JOINT_CLOSE = -0.6213

############################ Helper functions ############################

MASTER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_POSITION_CLOSE) / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
MASTER_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE) + MASTER_GRIPPER_POSITION_CLOSE
PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE
MASTER2PUPPET_POSITION_FN = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(MASTER_GRIPPER_POSITION_NORMALIZE_FN(x))

MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
PUPPET_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
MASTER2PUPPET_JOINT_FN = lambda x: PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(MASTER_GRIPPER_JOINT_NORMALIZE_FN(x))

MASTER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)

MASTER_POS2JOINT = lambda x: MASTER_GRIPPER_POSITION_NORMALIZE_FN(x) * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
MASTER_JOINT2POS = lambda x: MASTER_GRIPPER_POSITION_UNNORMALIZE_FN((x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE))
PUPPET_POS2JOINT = lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x) * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
PUPPET_JOINT2POS = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN((x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE))

MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE)/2
