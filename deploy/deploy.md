# AGX Aloha：WebSocket 推理部署（只改 Wrapper，不改 `policy_server`）

本文说明如何在**不修改** `eval_utils/policy_server.py` 的前提下，通过 **`socket_test_optimized_AR_aloha.py`（或你的副本）里的 Wrapper**，把真机 / 客户端发来的观测转成与 **训练时 `GrootSimPolicy` / `experiment_cfg` 完全一致** 的 dict，并拿回动作。

`WebsocketPolicyServer` 只做：**收 msgpack → 调 `policy.infer(obs)` → 回传数组**。`policy_server.py` 里关于 `exterior_image_*`、`(N,8)` 的文字是 **RoboArena + DROID 的示例说明**，**不是**运行时校验。Aloha / agx_aloha 的字段与维数由 **你的 Wrapper + 真机 client** 约定。

---

## 部署前必读：训练侧「唯一真值」

推理必须与下面三处 **键名、顺序、维数、分辨率** 一致（以你本地文件为准）：

| 来源 | 路径 | 用来核对什么 |
|------|------|----------------|
| 训练脚本 | `scripts/train/agx_aloha_training.sh` | `image_resolution_width=320`、`image_resolution_height=176`、`action_horizon=24`、`max_chunk_size=4` 等 |
| 模态与拼接顺序 | `groot/vla/configs/data/dreamzero/base_48_wan_fine_aug_relative.yaml` 中 `modality_config_agx_aloha` | **三路视频名及顺序**、`state` / `action` 子键列表、`annotation.task` |
| 你数据集上的维数 | `cobot_data/cobot_dataset/meta/modality.json`（或你的 `AGX_ALOHA_DATA_ROOT/meta/modality.json`） | 每个 `state.*` / `action.*` 在 **打包向量里的 index**，以及视频 **LeRobot 列名** |

**三路视频在训练里的顺序（拼接顺序即此列表顺序）：**

1. `video.cam_high`  
2. `video.cam_left_wrist`  
3. `video.cam_right_wrist`  

见 `base_48_wan_fine_aug_relative.yaml` 中 `modality_config_agx_aloha.video.modality_keys`。

**语言键（训练）：** `annotation.task`（不是 DROID 的 `annotation.language.action_text`）。

**Checkpoint：** 使用带权重的目录，例如  
`./checkpoints/agx_aloha_h100_run1/checkpoint-20000/`  
（内含 `config.json`、`model.safetensors`、`experiment_cfg/conf.yaml`、`experiment_cfg/metadata.json`）。

---

## Overview

```
Step 0  对齐「真值」：打开 modality.json + conf.yaml，写下来
Step 1  main()：embodiment_tag、model_path、PolicyServerConfig
Step 2  Wrapper：2.1 `_frame_buffers` 定义；2.2 `_convert_observation`；2.3 `_reset_state`；2.4 `_save_input_obs` 枚举 key
Step 3  Wrapper：线协议「相机槽位」→ 三个 video.*（语义与训练一致）
Step 4  Wrapper：时间维 T（首包 1 帧，之后每包 4 帧）
Step 5  Wrapper：分辨率（推荐 client 侧 resize；可选 server assert）
Step 6  Wrapper：state（四个 state.* 或 observation/proprio 切片）
Step 7  Wrapper：prompt → annotation.task
Step 8  Wrapper：_convert_action → (N, 14) 与真机约定一致
Step 9  其它硬编码：存图、注释、类名
Step 10 启动命令与真机 client 自检
```

---

## Step 0：先写一张「对齐表」（纸笔或注释在文件头）

**本仓库数据集真值文件路径（你当前工程）：**  
`/data/user/wsong890/shuaizhou/dreamzero/cobot_data/cobot_dataset/meta/modality.json`

下面表格已按该文件填好；若你换数据集，以**那份** `meta/modality.json` 为准。

### 视频（训练 / 推理 dict 里的 key → LeRobot 里存的列名）

| 训练 & 推理用 key | `modality.json` 中 `original_key`（parquet 侧） | 拼接顺序 |
|-------------------|-----------------------------------------------|----------|
| `video.cam_high` | `observation.images.cam_high` | 第 1 路 |
| `video.cam_left_wrist` | `observation.images.cam_left_wrist` | 第 2 路 |
| `video.cam_right_wrist` | `observation.images.cam_right_wrist` | 第 3 路 |

推理时送入 `GrootSimPolicy` 前，每个 key 对应 **`(T, 176, 320, 3)`**，**`uint8` RGB**（`T` 见 Step 4；空间分辨率与 `agx_aloha_training.sh` 中 `image_resolution_*` 一致）。

### State（四个子键都来自同一条打包向量 `observation.state`）

`modality.json` 里用 **`[start, end)`** 半开区间标在 **14 维** `observation.state` 上：

| 训练 & 推理用 key | `original_key` | index `[start, end)` | 每步形状 |
|-------------------|----------------|-------------------------|----------|
| `state.left_joint_pos` | `observation.state` | `[0, 6)` → **6** 维 | `(1, 6)` |
| `state.left_gripper_pos` | `observation.state` | `[6, 7)` → **1** 维 | `(1, 1)` |
| `state.right_joint_pos` | `observation.state` | `[7, 13)` → **6** 维 | `(1, 6)` |
| `state.right_gripper_pos` | `observation.state` | `[13, 14)` → **1** 维 | `(1, 1)` |

**真机线协议建议：** 发 **`observation/proprio`**，长度 **14**，**下标顺序与上表一致**：`[0:6]` 左臂关节、`[6]` 左夹爪、`[7:13]` 右臂关节、`[13]` 右夹爪（与 `observation.state` 一条向量顺序相同）。Wrapper 里再切成四个 `state.*` 赋给模型。

### Action（训练时 parquet 里打包列 `action`；推理输出再拼回 14 维）

| 子键名（模型 `action.*`） | `original_key` | index `[start, end)` |
|---------------------------|----------------|------------------------|
| `action.left_joint_pos` | `action` | `[0, 6)` |
| `action.left_gripper_pos` | `action` | `[6, 7)` |
| `action.right_joint_pos` | `action` | `[7, 13)` |
| `action.right_gripper_pos` | `action` | `[13, 14)` |

Wrapper 的 **`_convert_action`** 应按 **同一顺序** 拼成 **`(N, 14)`** 回给真机（`N` 一般为 `action_horizon`，如 24）。

### 语言

| 训练 & 推理用 key | `original_key` |
|-------------------|----------------|
| `annotation.task` | `annotation.task` |

Wrapper：`prompt`（或 client 自定义字段）→ **`annotation.task`** 字符串。

再打开 **`checkpoint-*/experiment_cfg/conf.yaml`**，确认：

- `image_resolution_width` / `image_resolution_height` 是否为 **320 / 176**  
- `embodiment_tag` 推理用 **`agx_aloha`**  
- `relative_action`、`relative_action_keys` 与训练一致  

**本 Step 在干什么：** 避免「线协议槽位名」和「训练时哪路相机」在脑子里对不上；后面 Wrapper 里每一行映射都应能在这张表上找到依据。

---

## Step 1：修改 `main()`（`socket_test_optimized_AR_aloha.py`）

### 1.1 `embodiment_tag`

- **改什么：** 设为 **`"agx_aloha"`**（字符串与 `EmbodimentTag.AGX_ALOHA` 一致）。
- **在干什么：** `GrootSimPolicy` 会加载 `experiment_cfg` 里 **agx_aloha** 的 `transforms` 和 `metadata.json` 中对应条目；写错 embodiment 会直接 assert 或归一化错。

### 1.2 `GrootSimPolicy(..., model_path=...)`

- **改什么：** `model_path` 指向 **具体 step 目录**，例如  
  `--model-path ./checkpoints/agx_aloha_h100_run1/checkpoint-20000`
- **在干什么：** 该目录下必须有 `experiment_cfg/` 与权重；LoRA 训练时 `save_lora_only=true` 的加载逻辑已在 `sim_policy.py` 里处理。

### 1.3 `PolicyServerConfig`（仍用 `eval_utils.policy_server`，只改参数）

- **改什么：**
  - `image_resolution`：若真机 **已在发送前** resize 到 **高 176 × 宽 320**，写 **`(176, 320)`**（tuple 表示 **高度, 宽度**，与原版 DROID 示例 `(180, 320)` 同一约定）。若你希望 client 仍发 **480×640** 仅作提示，可写 `(480, 640)`，但须在 Step 5 明确 **谁在 resize**。
  - `needs_wrist_camera=True`、`n_external_cameras=2`：与 **三路图** 的 RoboArena 线协议一致（两路算 “exterior”、一路算 “wrist”），**不要求**你真有「外置」物理结构。
  - `needs_session_id=True`：建议保留，便于新 episode 清缓存。
- **在干什么：** 连接建立后，服务器会把该 dict **发给 client**，用于配置 client 发什么；**不改变** `infer` 里实际收到的 key，那些仍由你的 Wrapper 解析。

### 1.4 `wrapper_policy = YourWrapperClass(...)`

- **改什么：** 使用你实现好的 Wrapper 类（见 Step 2–8），传入 `groot_policy`、`signal_group`、`output_dir`。
- **在干什么：** 只有 rank 0 走 WebSocket + Wrapper；worker rank 仍参与 `lazy_joint_forward_causal` 的分布式前向。

---

## Step 2：Wrapper 里所有和「三路 `video.cam_*`」相关的位置

**统一文件：** `socket_test_optimized_AR_aloha.py`（下文行号以 **当前仓库该文件** 为准；你本地改过就 `rg _frame_buffers socket_test_optimized_AR_aloha.py` 对一下。）

---

### 2.1 `AlohaPolicy.__init__` — 初始化 `_frame_buffers` 的三个 key

| 项 | 内容 |
|----|------|
| **文件** | `socket_test_optimized_AR_aloha.py` |
| **类** | `AlohaPolicy` |
| **方法** | `__init__` |
| **行号（参考）** | 约 **第 57–72 行**（搜 **`self._frame_buffers:`** 定位） |

**改什么：** 字典 **`self._frame_buffers`** 的三个 **key** 必须是（与训练一致）：

```text
"video.cam_high"
"video.cam_left_wrist"
"video.cam_right_wrist"
```

**在干什么：** 规定「每个逻辑相机一路」一个 list，后面 `_convert_observation` 只往这三个 key 里 `append` / `extend` 帧；最终 `converted` 里也要出现同名 `video.*` 供 `GrootSimPolicy` 使用。

**不要写成：** `video.exterior_image_*`、`video.wrist_image_left`；也不要只有两个 key。

---

### 2.2 `AlohaPolicy._convert_observation` — 按行改（`socket_test_optimized_AR_aloha.py` 约 **87–176** 行）

**定位：** 类 **`AlohaPolicy`**，方法 **`_convert_observation`**。下面行号以 **当前仓库** 该文件中 **`def _convert_observation`** 所在块为准（你增删过代码就整体平移对照）。

---

#### 2.2.1 第 **88–105** 行：函数 docstring

| 行号（约） | 现在是什么 | 你要做什么 |
|------------|------------|------------|
| **88–105** | 仍写 **Roboarena → AR_droid**、`exterior_image_*`、`joint_position(7)`、`annotation.language.action_text` | **整段删掉重写**。改成只描述 **你部署实际支持的 client 键** → **模型键**（`video.cam_*`、`state.*`、`annotation.task`），避免后人照注释去实现 DROID。 |

**在干什么：** 文档与真实逻辑一致；不参与运行。

---

#### 2.2.2 第 **108** 行：注释

| 行号（约） | 你要做什么 |
|------------|------------|
| **108** | 把「roboarena / droid」类注释改成：**「client 观测键 → `video.cam_*` buffer」**。 |

---

#### 2.2.3 第 **109–113** 行：`image_key_mapping` 字典

| 行号（约） | 现在是什么 | 你要做什么 |
|------------|------------|------------|
| **109–113** | 只有 `observation/cam_high` 等 **直接 cam 名** → `video.cam_*` | **二选一或合并**：<br>• **A**：若真机继续用 RoboArena 三路名发图，**必须增加**三行：`'observation/exterior_image_0_left' → video.cam_high`、`'observation/exterior_image_1_left' → video.cam_left_wrist`、`'observation/wrist_image_left' → video.cam_right_wrist`（顺序与 Step 0 物理相机一致，见 Step 3）。<br>• **B**：若真机永远只发 `observation/cam_*`，可保持现状，但 **不要**指望 `exterior_*` client 能工作。 |

**在干什么：** 决定 **哪一条 WebSocket 字段** 被 `append` 进 **哪一个** `_frame_buffers[...]` list；**右侧 value 必须是** `video.cam_high` / `video.cam_left_wrist` / `video.cam_right_wrist` 之一。

---

#### 2.2.4 第 **115–125** 行：`for roboarena_key, model_key in image_key_mapping.items()` 循环体

| 行号（约） | 现在是什么 | 你要做什么 |
|------------|------------|------------|
| **116** | `for roboarena_key, droid_key in ...` | 建议把 **`droid_key` 变量名改成 `model_key`**（避免误解成 DROID）；**逻辑可保留**。 |
| **117–118** | `if roboarena_key in obs` | **保留**。 |
| **119–125** | `extend` / `append` 到 `self._frame_buffers[droid_key]` | 把下标 **`droid_key` 全改成与 116 一致的 `model_key`**（即必须是 **`video.cam_*`**）。 |

**在干什么：** 把本包收到的 numpy 帧推进 **2.1** 定义的三个 buffer。

---

#### 2.2.5 第 **127–133** 行：`num_frames`

| 行号（约） | 你要做什么 |
|------------|------------|
| **127–133** | **一般不改**。首包 `1`、之后 `FRAMES_PER_CHUNK`（4）与 AR 推理一致（详见 Step 4）。 |

---

#### 2.2.6 第 **135–149** 行：从 buffer 堆 `video` 写入 `converted`

**一句话：** 把三个 **list 缓冲**里攒的帧 **`stack`** 成 **`(T,H,W,3)`**，写进 **`converted`**，键名必须是 **`video.cam_high` / `video.cam_left_wrist` / `video.cam_right_wrist`**。

---

**你要改什么？——分三种情况读：**

**情况 A（多数情况，可以不改逻辑）**  
- `_frame_buffers` 的 key 已经是上面三个 **`video.cam_*`**（Step 2.1 对了）。  
- 循环是：`for xxx, buffer in self._frame_buffers.items():` … `converted[xxx] = video`，**`xxx` 和 dict 的 key 是同一个东西**。  

→ **这一段的数学逻辑不用动**；最多把变量名 `droid_key` 改成 `model_key`，读着不误会。

**情况 B（必须改）**  
- 循环变量还是 `droid_key`，但 **`converted[droid_key]` 里实际会出现 `video.exterior_*` 等 DROID 名**，或 `_frame_buffers` 的 key 还没改成 `video.cam_*`。  

→ **必须**改成：遍历的 key 与写入 **`converted[...]`** 的 key **都只能是** 三个 **`video.cam_*`**（与 2.1、2.2.3 一致）。

**情况 C（可选，防 client 漏发某一相机）**  
- 某一 buffer 一直是空的 → 这个 `for` **不会**给 `converted` 加上对应 **`video.cam_*`** → 模型可能缺 key。  

→ 在 **`for ... in self._frame_buffers.items()` 整个循环结束后**（原「约 149 行」后面），对每个 **`video.cam_*`** 检查 **`if key not in converted`**，补 **`np.zeros((num_frames, 176, 320, 3), dtype=np.uint8)`**；其中 **`num_frames`** 用本函数里刚算好的 **`T`（1 或 4）**，**`176,320`** 与训练分辨率一致。  

→ **若你保证三路每步必发**，**不用做**情况 C。

---

**在干什么：** 得到 **`converted["video.cam_*"]`**，形状 **`(T,H,W,3)`**，给 `GrootSimPolicy`；**`T`** 由 2.2.5 决定。

---

#### 2.2.7 第 **151–168** 行：state（当前仍是 DROID 7+1，**必须改**）

| 行号（约） | 现在是什么 | 你要做什么 |
|------------|------------|------------|
| **151–168** | `observation/joint_position` / `gripper` → `state.joint_position` / `state.gripper_position` | **整段删除**，换成 **agx_aloha** 四键逻辑（与 Step 0 `modality.json` 的 `[start,end)` 一致）：<br>• 读 **`observation/proprio`** `(14,)`（或你约定的键），按 **0:6、6:7、7:13、13:14** 切成 **`state.left_joint_pos` `(1,6)`**、**`state.left_gripper_pos` `(1,1)`**、**`state.right_joint_pos` `(1,6)`**、**`state.right_gripper_pos` `(1,1)`**；<br>• 或读四个 `observation/...` 分键再 `reshape`；<br>• **禁止**再写 **`state.joint_position` / `state.gripper_position`**。 |

**在干什么：** 与训练 **`observation.state` 拆列方式** 一致，否则 `eval_transform` + 相对动作 `unapply` 会错。

---

#### 2.2.8 第 **170–174** 行：语言

| 行号（约） | 现在是什么 | 你要做什么 |
|------------|------------|------------|
| **170–174** | `converted["annotation.language.action_text"] = obs["prompt"]` | **改成** **`converted["annotation.task"] = str(obs.get("prompt", ""))`**（或你 client 实际用的指令字段名，但 **左侧 key 必须是 `annotation.task`**）。**删掉**对 **`annotation.language.action_text`** 的赋值。 |

**在干什么：** 与训练 **`annotation.task`** 一致（见 `modality.json` → `annotation.task`）。

---

#### 2.2.9 第 **176** 行

| 行号（约） | 你要做什么 |
|------------|------------|
| **176** | **`return converted`**：**保留**。返回给 `infer` → `Batch(obs=...)` 的 dict。 |

---

**2.2 小结（检查清单）：**

1. **109–125、136–149**：三路图最终只出现 **`video.cam_*`** 三种 key。  
2. **151–168**：只出现 **四个 `state.*`（agx）**，维数与 Step 0 表一致。  
3. **170–174**：只有 **`annotation.task`**，没有 `annotation.language.action_text`。

---

### 2.3 `AlohaPolicy._reset_state` — 清空 `_frame_buffers`

| 项 | 内容 |
|----|------|
| **类** | `AlohaPolicy` |
| **方法** | `_reset_state` |
| **行号（参考）** | 约 **第 346–347 行**（搜 **`for key in self._frame_buffers`**） |

**改什么：** 一般 **不用改逻辑**：`for key in self._frame_buffers: self._frame_buffers[key] = []` 会按 2.1 里定义的三个 key 清空。只要 2.1 的 key 对了，这里自动对。

**在干什么：** 新 session / reset 时丢掉历史帧，避免跨 episode 混帧。

---

### 2.4 `WebsocketPolicyServer._save_input_obs` — 存调试图时枚举的 key

| 项 | 内容 |
|----|------|
| **文件** | 仍是 `socket_test_optimized_AR_aloha.py`（本文件里 **自写了一份** `WebsocketPolicyServer`，不是 `eval_utils/policy_server.py`） |
| **类** | `WebsocketPolicyServer` |
| **方法** | `_save_input_obs` |
| **行号（参考）** | 约 **第 407 行** 的 **`for key in (...)`** 元组 |

**改什么：** 把元组里的三个字符串从 DROID 的  

`video.exterior_image_1_left`、`video.exterior_image_2_left`、`video.wrist_image_left`  

改成 **`video.cam_high`、`video.cam_left_wrist`、`video.cam_right_wrist`**（与 2.1 一致）。否则 server 存 debug 图时 **找不到** `converted` 里的 agx 键，或仍按旧路径存空。

**在干什么：** 仅影响 **落盘调试 PNG**；不影响 `infer` 正确性，但排障时容易误导。

---

### Step 2 总目的（对照检查）

- 全文件搜 **`_frame_buffers`**、**`video.exterior`**、**`wrist_image`**：除历史注释外，**参与推理与存图的路径**上不应再出现 DROID 视频 key。  
- 名称必须与 **`base_48_wan_fine_aug_relative.yaml` → `modality_config_agx_aloha.video`** 以及 **`meta/modality.json` → `video`** 一致。

---

## Step 3：只改一处字典 — 把「client 发来的图字段名」接到 `video.cam_*`

### 3.0 你实际要做什么（一句话）

在 **`socket_test_optimized_AR_aloha.py`** 里，找到 **`AlohaPolicy._convert_observation`** 中的变量 **`image_key_mapping`**（当前约在 **第 112–116 行**），把它改成 **「左侧 = WebSocket 里可能出现的 key」「右侧 = 只能是三个 `video.cam_*`」**。  
**下面 `for roboarena_key, model_key in image_key_mapping.items():` 那一整段循环不要删、不要另起函数**——它已经会把 `obs` 里的图塞进 `_frame_buffers[model_key]`。

---

### 3.1 改哪个文件、哪一段

| 项 | 值 |
|----|-----|
| 文件 | `socket_test_optimized_AR_aloha.py` |
| 类 | `AlohaPolicy` |
| 方法 | `_convert_observation` |
| 变量名 | **`image_key_mapping`** |
| 位置 | **`converted = {}` 下面**，紧挨着 **`for roboarena_key, model_key in image_key_mapping.items():`** 上面（约 **111–128** 行一带） |

---

### 3.2 推荐你直接用的字典（可复制整块替换 `image_key_mapping = { ... }`）

含义（须与 Step 0 / 你真实相机一致；若你左右腕对调了采集，就改右侧三行映射）：

| client 发来的 key（左） | 接到模型的 buffer / converted key（右） |
|-------------------------|----------------------------------------|
| `observation/cam_high` 或 `observation/exterior_image_0_left` | `video.cam_high`（= 训练时 `cam_high` 那路） |
| `observation/cam_left_wrist` 或 `observation/exterior_image_1_left` | `video.cam_left_wrist` |
| `observation/cam_right_wrist` 或 `observation/wrist_image_left` | `video.cam_right_wrist` |

**复制用代码（与仓库当前实现一致时可核对）：**

```python
        image_key_mapping = {
            "observation/cam_high": "video.cam_high",
            "observation/cam_left_wrist": "video.cam_left_wrist",
            "observation/cam_right_wrist": "video.cam_right_wrist",
            "observation/exterior_image_0_left": "video.cam_high",
            "observation/exterior_image_1_left": "video.cam_left_wrist",
            "observation/wrist_image_left": "video.cam_right_wrist",
        }
```

**说明：** 同一路 **物理相机** 只允许 client **二选一** 发键名（例如只发 `cam_high` 或只发 `exterior_image_0_left`），不要两键同时塞**同一帧**重复进 buffer；若都发，循环会 **append 两次** 同一路。

---

### 3.3 你**不需要**做的事

- **不要**新建文件、**不要**在 Step 3 单独再写第二个 `for` 去拷图；**只改** `image_key_mapping` 即可。  
- **`policy_server.py` 不用改**；RoboArena 文档里的 `exterior_*` 只是「常见 client 键名」，真正接哪路相机由上面字典右列决定。

---

### 3.4 和训练「顺序」的关系

训练里三路拼接顺序是 **`cam_high` → `cam_left_wrist` → `cam_right_wrist`**。  
字典 **不决定时间顺序**，只决定 **哪包 numpy 进哪个 `video.*`**；**顺序对错**取决于你是否把 **真机 high 的图** 接到了 **`video.cam_high`**（而不是接到 `cam_left_wrist`）。若接反了，就 **交换字典里某两行的右侧**（或改 client 发哪一路进哪个槽）。

---

## Step 4：Wrapper — 时间维 `T`（与 AR 推理脚本一致）

- **改什么：** 保留与 `socket_test_optimized_AR.py` 相同逻辑即可：
  - **该 `session_id` 下第一次 `infer`：** 每个相机用 **最后 1 帧** 组成 `T=1`；
  - **同 session 后续每次 `infer`：** 每个相机用 **最后 4 帧** 组成 `T=4`（`FRAMES_PER_CHUNK = 4`）。
- **在干什么：** 与当前 DreamZero AR 分布式推理里 `lazy_joint_forward_causal` 的用法一致；与训练里 **eval 视频时间索引**（`eval_delta_indices: [-3,-2,-1,0]`，共 4 个时间步）在语义上对齐。**不要**在未改模型的情况下自行改成「每步只发 1 帧」除非你也改 action head 的 chunk 逻辑。

---

## Step 5：Wrapper — 分辨率

- **推荐（你已选择）：** 真机 / client 在 **发送前** 把每路图 resize 到 **宽 320 × 高 176**，`uint8`，形状 **`(H, W, 3)`** 或 **`(T, H, W, 3)`**。
- **改什么：** Wrapper 里可写 **`assert`**：对每个 `video.*`，`arr.shape[-3] == 176` 且 `arr.shape[-2] == 320`（注意 numpy 是 **…, H, W, C**）。
- **在干什么：** 与 `VideoResize` 在训练配置里使用的 `image_resolution_height` / `width` 一致；避免 silent 错误。若你坚持 server 再 resize 一层也可以，但会与「client 已处理」重复，二选一即可。

---

## Step 6：Wrapper — `state` 四个键

训练期望四个键（名称固定，维数见 `modality.json`）：

- `state.left_joint_pos`
- `state.left_gripper_pos`
- `state.right_joint_pos`
- `state.right_gripper_pos`

**改什么（两种常见做法，二选一或都支持）：**

1. **真机发一个向量 `observation/proprio`，形状 `(14,)`（或 `(1,14)`）**  
   - 在 Wrapper 里按 **Step 0 表中 `[start, end)`** 切成四段：`6+1+6+1`，每段 `reshape(1, d)` 写入四个 `state.*`（与 `cobot_dataset/meta/modality.json` 中 `state` 段一致）。

2. **真机分开发四个 `observation/...` 键**  
   - Wrapper 里分别 `reshape(1, d)` 赋给四个 `state.*`。

**在干什么：** `sim_policy.py` 里 `relative_action` 为 true 时，`unapply` 会按 **子键名** 把相对动作加回 **同名 state**；state 键名或维数错了，动作会错。

---

## Step 7：Wrapper — 语言

- **改什么：** `converted["annotation.task"] = str(obs.get("prompt", ""))`（或你约定的 client 字段名）。
- **在干什么：** 与训练时 `modality_config_agx_aloha.language.modality_keys` 一致；不要用 `annotation.language.action_text`。

---

## Step 8：Wrapper — `_convert_action`

- **改什么：** 从 `result_batch.act` 里取出带 **`left_joint_pos` / `left_gripper_pos` / `right_joint_pos` / `right_gripper_pos`** 的项（注意 Tianshou `Batch` 里 key 的字符串形式），按 **与真机控制器约定** 的顺序拼成 **`numpy` 形状 `(N, 14)`**（若总维不是 14，以 `modality.json` 为准）。
- **在干什么：** WebSocket 回包给真机执行；真机按同一顺序拆成左右臂指令。原先 DROID 的 **`(N, 8)`** 必须删除。

---

## Step 9：全局搜索 DROID 残留

在 **`socket_test_optimized_AR_aloha.py`** 内全文搜索并替换/删除：

| 搜索串 | 处理 |
|--------|------|
| `exterior_image_1_left`、`wrist_image_left`（作为 **输出 dict 的 key**） | 不应再出现在 `converted` 里 |
| `state.joint_position`、`state.gripper_position` | 不应再出现 |
| `annotation.language.action_text` | 不应再出现 |
| `_save_input_obs` 等调试存图里的 key 列表 | 改为三个 `video.cam_*` |

**在干什么：** 防止 debug 路径或注释仍指向旧 DROID 键，和当前 Wrapper 行为不一致。

---

## Step 10：启动与真机自检

### 10.1 启动 server（示例）

与 README 类似，**路径与 GPU 数按你机器修改**：

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc_per_node=2 \
  socket_test_optimized_AR_aloha.py \
  --port 5000 \
  --enable-dit-cache \
  --model-path ./checkpoints/agx_aloha_h100_run1/checkpoint-20000
```

### 10.2 真机 client 每条 `infer` 建议携带

| 字段 | 类型 / 形状 | 说明 |
|------|----------------|------|
| `endpoint` | `"infer"` | `eval_utils/policy_client.py` 会自动加；自写 client 时勿漏 |
| `session_id` | 可序列化值 | 新任务换新 id，便于清 buffer |
| 三路图像 | 见 Step 3–5 | 键名按你 Wrapper 映射表；**首步单帧、后续每步 4 帧** |
| `observation/proprio` | `(14,)` float | 顺序与 `modality.json` 一致；或发四个分键 |
| `prompt` | `str` | 映射到 `annotation.task` |

### 10.3 最小自检

1. 第一次 `infer` 后，在 Wrapper里 **log 三个 `video.*` 的 `shape`**，应为 **`(1,176,320,3)`** 或 **`(4,176,320,3)`**。  
2. 打印四个 `state.*` 的 `shape`，与 `modality.json` 一致。  
3. 返回 `action.shape == (N, 14)`（N 与 `action_horizon` 一致，一般为 24）。

---

## 常见问题（简短）

**Q：`policy_server.py` 文档和我不一样，要不要 fork？**  
A：不必。文档是示例；约束在 **Wrapper 输出的 dict** 与 **checkpoint 的 experiment_cfg**。

**Q：我只有「一个 high」，没有两路 exterior？**  
A：线协议仍占三个槽；第三路可以是右腕或重复视角，**以训练数据列名为准**，不能随意缺一路不填（缺则需在 Wrapper 里显式补零或重复，并清楚这会引入分布偏移）。

**Q：能否改 `eval_utils/policy_server.py`？**  
A：可以只改 docstring 注明「以具体 Policy 为准」；**功能上不必改**。

---

## 相关文件（便于跳转）

| 文件 | 作用 |
|------|------|
| `socket_test_optimized_AR_aloha.py` | 分布式推理入口 + **你的 Wrapper** |
| `eval_utils/policy_server.py` | 通用 WebSocket server（一般不改） |
| `eval_utils/policy_client.py` | Python 参考 client |
| `groot/vla/model/n1_5/sim_policy.py` | `GrootSimPolicy` 加载 checkpoint、merge LoRA、`eval_transform` |
| `scripts/train/agx_aloha_training.sh` | 训练分辨率与 horizon 等 CLI 默认值 |

---

*若你之后把 Wrapper 固化为一版「与 cobot_dataset 完全一致」的实现，可在本文件顶部加一行：「以 commit & 数据集路径为真值」，便于复现。*
