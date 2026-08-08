# ACT CVAE 架构原理详解

> 基于 Mobile ALOHA 论文 + ACT-plus-plus 代码分析  
> 日期：2026-04-20

---

## 一、网络结构总览

ACT（Action Chunking with Transformers）的核心是一个 **条件变分自编码器（CVAE）**，由两个独立网络组成：

```
┌─────────────────────────────────────────────────────────────────────┐
│  ① CVAE Encoder (self.encoder)                                      │
│     结构: TransformerEncoder (仅encoder层，无decoder)                 │
│     参数: cls_embed, encoder_action_proj, encoder_joint_proj,        │
│           latent_proj                                                │
│     输入: [CLS] + qpos + a₁..aₖ  （只有关节状态和GT动作）              │
│     输出: μ, log σ² → 重参数化 → z (32维)                             │
│     ⚠️ 与图像完全无关                                                 │
├─────────────────────────────────────────────────────────────────────┤
│  ② 主网络 / CVAE Decoder (self.transformer + backbones)              │
│     结构: ResNet18 backbone + 完整 Transformer (encoder+decoder)     │
│     参数: backbones, input_proj, transformer, action_head,           │
│           latent_out_proj, query_embed                               │
│     输入: 图像特征 + qpos + z                                         │
│     输出: â₁..âₖ (预测动作序列)                                       │
│     ⚠️ 这里才用到图像                                                 │
└─────────────────────────────────────────────────────────────────────┘
```

关键区分：
- **CVAE Encoder**：只看动作序列和关节状态，**不看图像**
- **主网络（CVAE Decoder）**：看图像、关节状态和潜变量 z，**不看GT动作**

---

## 二、Training（训练）全流程

对应代码 `policy.py` 第209行 `actions is not None` 分支。

输入数据：`qpos`（关节状态）、`image`（3个相机）、`actions`（GT动作序列）、`is_pad`（padding掩码）

### Step 1 — CVAE Encoder 生成 z

对应代码 `detr/models/detr_vae.py` 第97-126行：

```
actions ──→ encoder_action_proj ──→ action_embed  (bs, k, 512)
qpos   ──→ encoder_joint_proj  ──→ qpos_embed    (bs, 1, 512)
                                    cls_embed      (bs, 1, 512)
                                        ↓
拼接: [CLS, qpos, a₁, a₂, ..., aₖ]  → (k+2, bs, 512)
                                        ↓
              TransformerEncoder (4层 self-attention)
                                        ↓
              取 CLS 输出 → latent_proj → [μ, log σ²] (各32维)
                                        ↓
              重参数化: z = μ + σ·ε     (ε ~ N(0,1))
                                        ↓
              latent_out_proj(z)  → latent_input (512维)
```

**关键**：这一步**完全不看图像**。Encoder 只需要知道"这条轨迹的动作序列长什么样"，把动作的"风格/意图"压缩到 32 维 z 里。

### Step 2 — 主 Transformer 预测动作

对应代码 `detr/models/detr_vae.py` 第138-162行：

```
image ──→ ResNet18 backbone ──→ CNN特征图 (bs, 512, h, w)
      ──→ input_proj (1×1 conv) ──→ src   (bs, 512, h, w×3相机拼接)
                                      + 位置编码 pos

qpos  ──→ input_proj_robot_state ──→ proprio_input (bs, 512)

z     ──→ (来自 Step 1)           ──→ latent_input  (bs, 512)

               ┌─────────────────────────────────┐
               │ Transformer Encoder              │
               │  输入: src (图像特征序列)          │
               │  self-attention → 图像理解        │
               ├─────────────────────────────────┤
               │ Transformer Decoder              │
               │  query: query_embed (k个可学习)   │
               │  额外token: [proprio, latent_z]  │
               │  cross-attention → 看图像特征     │
               │  输出: hs (bs, k, 512)           │
               └─────────────────────────────────┘
                              ↓
               action_head(hs) → â₁..âₖ  (bs, k, 16)
```

### Step 3 — 计算 Loss

对应代码 `policy.py` 第220-234行：

$$\mathcal{L} = \underbrace{L_1(\hat{a}, a_{\text{GT}})}_{\text{动作重建损失}} + \beta \cdot \underbrace{D_{KL}(q(z|s,a) \| \mathcal{N}(0,I))}_{\text{KL正则}}$$

- **L1 loss**：预测动作 $\hat{a}$ 与 GT 动作的差距 → 让主网络学会从图像预测正确动作
- **KL loss**：迫使 z 分布趋近 N(0,I) → 推理时可以直接用 z=0 代替
- β（`kl_weight`）默认值为 10，KL 项只是轻微约束 z 不要偏离太远，但不会完全坍缩为零

---

## 三、Testing（推理）全流程

对应代码 `policy.py` 第235-237行 `actions is None` 分支。

输入：只有 `qpos` 和 `image`（没有 GT 动作）

### Step 1 — 直接令 z = 0

对应代码 `detr/models/detr_vae.py` 第128-133行：

```python
# 没有 actions → 无法运行 Encoder
latent_sample = torch.zeros([bs, self.latent_dim])  # z = 全零向量
latent_input = self.latent_out_proj(latent_sample)   # 投影到 512 维
```

**Encoder 整体跳过**，不调用任何 Encoder 参数。

### Step 2 — 主 Transformer 预测动作（与训练 Step 2 完全相同）

```
image ──→ ResNet18 ──→ Transformer Encoder ──→ 图像特征
qpos  ──→ proprio_input
z=0   ──→ latent_input
              ↓
     Transformer Decoder → â₁..âₖ
```

---

## 四、训练和推理之间的联系

### 连接物：Checkpoint（模型权重）

```
checkpoint 包含的权重:
├── ResNet18 backbone 权重    ← 训练时学会"怎么从图像提取特征"
├── Transformer Encoder 权重  ← 训练时学会"怎么理解图像空间关系"
├── Transformer Decoder 权重  ← 训练时学会"根据图像+qpos+z → 动作"
├── action_head 权重          ← 训练时学会"隐状态 → 16维关节角度"
├── latent_out_proj 权重      ← 把 z 投影到 512 维（训练和推理都用）
├── query_embed 权重          ← 可学习的动作查询向量
│
├── CVAE Encoder 权重         ← ⚠️ 存在 checkpoint 里，但推理时不调用
├── encoder_action_proj 权重  ← ⚠️ 同上
├── encoder_joint_proj 权重   ← ⚠️ 同上
├── latent_proj 权重          ← ⚠️ 同上
└── cls_embed 权重            ← ⚠️ 同上
```

推理时加载整个 checkpoint，但 CVAE Encoder 相关的权重**虽然加载了却从不被调用**（代码走 `else` 分支直接 z=0）。真正起作用的是**主 Transformer + backbone** 的权重——它们在训练中学到了"看图出动作"的能力。

---

## 五、核心问题：推理时 z=0，训练时为何要 CVAE Encoder？

### 问题本质：多模态平均（Mode Averaging）

假设同一个场景，人类演示了两种抓取方式：

```
场景: 积木在桌面中央
  演示 A: 先左手抓 → 放左槽 → 右手抓 → 放右槽
  演示 B: 先右手抓 → 放右槽 → 左手抓 → 放左槽
```

#### 没有 CVAE Encoder（直接 z=0 训练）

主网络面对**相同的图像输入**，却要拟合**两条完全不同的动作轨迹**：

$$\text{同一个 } f(\text{image}, \text{qpos}) \text{ 要同时输出 A 和 B}$$

网络无法做到，只能取**平均**：

$$\hat{a} = \frac{a_A + a_B}{2}$$

→ 一个两边都不像的错误动作，机器人可能撞到桌面或卡住。

#### 有 CVAE Encoder

```
训练时:
  演示 A 的动作序列 → Encoder → z_A ≈ [+0.8, -0.3, ...]
  演示 B 的动作序列 → Encoder → z_B ≈ [-0.5, +0.7, ...]

  主网络看到的是:
    f(image, qpos, z_A) → 学习输出 A 的动作  ✅ 不冲突
    f(image, qpos, z_B) → 学习输出 B 的动作  ✅ 不冲突
```

**z 为主网络消除了歧义！** 同一张图像 + 不同的 z = 不同的动作输出，网络不再被迫取平均。

### 推理时 z=0 为何可行？

1. **KL 正则**把 z 的分布压向 N(0,I)，z=0 位于分布中心，对应**最高概率的执行方式**（数据集中最常见的那种）
2. 对于机器人任务来说，**大多数场景只有一种合理方式**。多模态只在少数歧义场景出现
3. 即使出现歧义，z=0 输出的"最常见方式"通常也能完成任务

### 直觉类比：考试类比

- **训练阶段**：老师（Encoder）站在旁边，每道题都告诉学生（主网络）"这道题的解题思路类型是 z"。学生因此能**分门别类**地学习不同题型，不会把不同解法搞混
- **考试阶段**：老师不在了（z=0）。但学生已经把**最常用的解题套路**学得很扎实了（因为训练时没有被平均化干扰），直接用默认套路答题
- **如果训练时就没有老师**：学生面对相似的题看到矛盾的答案，学得一塌糊涂，考试时表现更差

---

## 六、总结

```
┌─────────────────────────────────────────────────────────────┐
│ CVAE Encoder 的真正角色：训练时的"辅助教练"                    │
│                                                             │
│ 训练时：                                                     │
│   Encoder 读GT动作 → 生成 z → 消除多模态歧义                  │
│   → 主网络能清晰地学习 image→action 的映射                    │
│   → 权重保存到 checkpoint                                    │
│                                                             │
│ 推理时：                                                     │
│   加载 checkpoint 中主网络的权重                               │
│   z=0（用最常见的风格）                                       │
│   主网络已经学好了"看图出动作"的能力 → 直接用                  │
│                                                             │
│ 连接物：checkpoint 中主网络的权重                              │
│ Encoder 的价值：让这些权重学得更好（不被多模态污染）            │
└─────────────────────────────────────────────────────────────┘
```

| 模块                        | 训练时   | 推理时   | 输入                   |
|------                      |-------- |-------- |------                  |
| **CVAE Encoder**           | ✅ 使用  | ❌ 丢弃 | 当前状态 + **GT动作序列** |
| **主网络（CVAE Decoder）**   | ✅ 使用 | ✅ 使用  | **图像** + 当前状态 + z  |

### 关键代码位置

| 内容 | 文件 | 行号 |
|------|------|------|
| ACTPolicy 训练/推理分支 | `policy.py` | 209-237 |
| CVAE Encoder (encode方法) | `detr/models/detr_vae.py` | 93-135 |
| 主网络 forward (图像处理+Transformer) | `detr/models/detr_vae.py` | 137-166 |
| KL 散度计算 | `policy.py` | 224 |
| L1 动作损失 | `policy.py` | 229-230 |
