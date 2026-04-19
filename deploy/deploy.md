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
| 你数据集上的维数 | `<数据集>/meta/modality.json` | 每个 `state.*` / `action.*` 的 **shape / index**，以及三路视频在数据里的 **列名** |

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
Step 2  Wrapper：帧缓冲字典的 key（必须是三个 video.cam_*）
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

在做任何代码修改前，打开你的 **`meta/modality.json`**，填下面表格（示例维数需你按文件改）：

| 训练用 key | 含义（你自己的相机/关节名） | 维数 / 形状 |
|------------|------------------------------|-------------|
| `video.cam_high` | 采集时对应哪一路物理相机 | 推理时 `(T, 176, 320, 3)`，`uint8` |
| `video.cam_left_wrist` | … | 同上 |
| `video.cam_right_wrist` | … | 同上 |
| `state.left_joint_pos` | … | `(1, d1)` |
| `state.left_gripper_pos` | … | `(1, d2)` |
| `state.right_joint_pos` | … | `(1, d3)` |
| `state.right_gripper_pos` | … | `(1, d4)` |
| `annotation.task` | 与训练时 task 文本同源 | `str` |

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

## Step 2：Wrapper 类 — `__init__` 里 `_frame_buffers` 的 key

- **改什么：** 三个 buffer 的 key 必须是（与训练 `modality_keys` 一致）：
  - `"video.cam_high"`
  - `"video.cam_left_wrist"`
  - `"video.cam_right_wrist"`
- **在干什么：** 后续 `_convert_observation` 输出的 dict 里必须出现这三个 **字符串 key**；`eval_transform` 按配置对它们做 crop/resize/normalize/concat。**顺序**由你「往哪个 buffer 里 append 哪路相机」决定，必须和 Step 0 表里「训练时哪路物理相机」一致，而不是和 `exterior` 字面一致。

---

## Step 3：Wrapper — 线协议「相机槽位」→ `video.cam_*`

### 3.1 为什么要映射

真机 WebSocket 仍可使用 RoboArena 风格键名（避免大改 client），例如：

- `observation/exterior_image_0_left`
- `observation/exterior_image_1_left`
- `observation/wrist_image_left`

或直接使用：

- `observation/cam_high`
- `observation/cam_left_wrist`
- `observation/cam_right_wrist`

**改什么：** 在 Wrapper 里维护一张 **Python 字典**：`client_key → 训练用 model_key`（右侧必须是 Step 2 三个 `video.cam_*` 之一）。

**在干什么：** 把「传输用名字」映射到 **训练时 parquet / modality 里的语义**。例如：若训练时 `cam_high` 列存的是固定俯视相机，则 **无论** 线协议里叫 `exterior_image_0` 还是 `cam_high`，**最终** `converted["video.cam_high"]` 里必须是这一路图像。

### 3.2 顺序如何「对上训练」

训练里的 **拼接顺序** 固定为：`cam_high` → `cam_left_wrist` → `cam_right_wrist`（见 YAML）。  
部署时要求：

- `converted["video.cam_high"]` 张量 = 训练时 **同名列** 的含义  
- 另外两个 key 同理  

**线协议 index 0/1/wrist 只是槽位**：你可以在注释里写死：「0 = high、1 = left_wrist、wrist = right_wrist」，但必须与 **Step 0 表** 一致。

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
   - 在 Wrapper 里按 **`modality.json` 里各子键的 index** 切成四段，每段 `reshape(1, d)` 写入上面四个 `state.*`。  
   - 若你当前代码用固定 `(6,1,6,1)` 切片，**必须与** 数据集里 **left/right 关节维数** 一致；不一致就改切片元组。

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
