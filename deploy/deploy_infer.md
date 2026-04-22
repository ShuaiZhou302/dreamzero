# DreamZero 真机推理客户端：`dz_aloha_infer.py` 改造说明

本文约定：**不讨论** conda / ROS 包 / `cv_bridge` 等依赖是否安装，默认环境已就绪。目标是把 `deploy/dz_aloha_infer.py` 改成 **ROS 收真机观测 → WebSocket 调 DreamZero server → 收 action → 再 publish 给机器人** 的闭环；模型侧与 **`socket_test_optimized_AR_aloha.py` + `AlohaPolicy`** 及 **`cobot_data/cobot_dataset/meta/modality.json`** 一致。

**ROS 侧「怎么订阅 / 同步帧 / 往真机发关节」不要凭记忆猜接口**：以仓库里的 **`deploy/other_vla_demo_deploy/demo_inference.py`** 为准（`RosOperator`、`get_frame()`、`puppet_arm_publish` / `puppet_arm_publish_continuous`、`JointState` topic 等）。当前 `dz_aloha_infer.py` 里为瘦身已去掉 **本地 ACT 推理** 大段代码；**实现 DreamZero infer 循环时**，把「原 demo 里接在 `policy(...)` 后面的 publish 逻辑」接到 **`client.infer(obs)` 返回的 `(N,14)`** 上即可，**ROS 发布接口仍与 demo 对齐**（仅把动作来源从本地模型换成 server 回包，维数按 agx 14 与 `modality.json` 拆左右臂）。

---

## 总览：改完后的文件职责

| 模块 | 保留 / 删除 / 新增 |
|------|-------------------|
| **`RosOperator`**（订阅图、关节状态；`get_frame()` 时间对齐；`puppet_arm_publish` 等） | **保留**结构与 topic 参数化；内部逻辑可小幅改名，**不必**为 DreamZero 重写同步算法 |
| **`ACTPolicy` / `make_policy` / `get_model_config` / `torch.load` / `stats` 归一化 / `inference_process` 线程里本地 `policy(...)`** | **删除**或整段不用 |
| **WebSocket 客户端** | **新增**：推荐复用 `eval_utils/policy_client.py` 的 `WebsocketClientPolicy(host, port)`，或等价自写 `msgpack_numpy` + `websockets` |
| **主循环 `model_inference` 一类** | **重写**为：`get_frame()` → 组 `obs` dict → `client.infer(obs)` → 解析返回值 → `puppet_arm_publish(...)` |

---

## Step 0：先对齐「真值」（改代码前读一遍）

与 server 侧 **`deploy/deploy_server.md`**、**`modality.json`** 一致：

- **图像**：三路语义为 `cam_high`、`cam_left_wrist`、`cam_right_wrist`；发送给 server 的 **线协议 key** 与 server 里 `image_key_mapping` 一致（例如 `observation/cam_high` …，或 `observation/exterior_image_0_left` 等）。
- **空间分辨率**：每路 **`uint8`**，**高 176 × 宽 320**，numpy 形状 **`(H, W, 3)`**（常见）或与 server 约定一致；发送前在 client 侧 **`cv2.resize`** 即可。
- **本体状态**：**`observation/proprio`**，**一维长度 14**，`float64/float32` 均可，顺序严格对应 `modality.json` 的 `state` 段：
  - `[0:6]` → 左臂 6 关节  
  - `[6]` → 左夹爪  
  - `[7:13]` → 右臂 6 关节  
  - `[13]` → 右夹爪  
- **语言**：obs 里带 **`prompt`**（`str`），server 侧映射为 **`annotation.task`**。
- **会话**：**`session_id`**（任意可序列化、每局任务建议唯一），便于 server 清视频 buffer；新任务换新 id。
- **时间维 T**（按 `test_client_AR.py` 风格固定）：**该 `session_id` 下第一次 `infer`** 每路发送 **1 帧**（HWC）；**同 session 后续每次 `infer`** 每路发送 **4 帧**（THWC）。server 侧与此语义对齐（首步 `T=1`、后续 `T=4`）。

**动作回包（目标形状）**：server 在 **`AlohaPolicy._convert_action` 按 agx 改完后** 应返回 **`numpy`，`shape == (N, 14)`**，`N` 与训练 **`action_horizon`** 一致（常见 **24**）。**14 维顺序与 `modality.json` 的 `action` 段相同**（与 `state` 分段一致：左 6+1，右 6+1）。若你当前仓库 server 仍返回 **`(N, 8)`**，客户端需暂时兼容或先完成 server Step 8 再联调。

---

## Step 1：文件头、`import`、全局变量

**删或不再使用：**

- `torch`（若仅用于 ACT）、`pickle`、`einops.rearrange`（若改为自己 stack 图像）、`from policy import ...`、`from utils import ...`（若不再本地训练/推理）。
- 全局 **`inference_thread` / `inference_lock` / `inference_actions` / `inference_timestep`** 以及 **`actions_interpolation`**、**`inference_process`** 中与 **本地 GPU 推理** 绑定的逻辑。

**增加：**

- `from eval_utils.policy_client import WebsocketClientPolicy`（或你把 `policy_client` 拷到 `deploy/` 下的等价路径）。
- 如需显式打包： `from openpi_client import msgpack_numpy`（与 server 一致）。

**可保留：**

- `argparse`、`numpy`、`collections` / `deque`、`rospy`、几何与传感器消息、`CvBridge`、`time`、`threading`（若仍用独立线程做 infer，可选；第一版可在主线程同步 `infer`）。

---

## Step 2：删除 ACT 专用配置块

**删除**（或整文件替换后不再出现）：

- **`task_config`** 里若仅给 ACT 用、且与 ROS 相机 topic 无绑定的可删；若仍用作文档说明「front→high」的语义表，可改成 **常量字典** `ROS_CAMERA_TO_OBSKEY = {...}`，见 Step 6。
- **`get_model_config` / `make_policy` / `get_image` / `get_depth_image`** 等 **仅为本地 ACT** 服务的函数。

若仍要 **深度图 / 底盘**：DreamZero 当前 agx_aloha server 未在文档中要求这些键，**第一版建议关深度、关 base**，与 server 一致后再扩展。

---

## Step 3：`argparse`（`get_arguments`）建议参数

在原有 **ROS topic、`publish_rate`** 等基础上 **增加**：

| 参数 | 含义 |
|------|------|
| `--server-host` | DreamZero WebSocket server 地址（如 `127.0.0.1` 或调度节点 IP） |
| `--server-port` | 端口，与 `socket_test_optimized_AR_aloha.py --port` 一致 |
| `--task-name` 或 `--prompt` | 自然语言任务描述，写入 obs **`prompt`**；可与「任务配置文件名」混用，但发给 server 的必须是 **字符串 `prompt`** |
| `--session-id` | 可选；若不传，程序启动时用 **`uuid` 或时间戳** 生成一个，整段 infer 复用；**新开一局任务**时换新 id |
| `--image-height` / `--image-width` | 默认 **176**、**320**，与训练一致 |

保留 demo 里 **`img_front_topic` / `img_left_topic` / `img_right_topic`**、`puppet_arm_*`、`publish_rate` 等，便于真机换线不改代码。

---

## Step 4：`RosOperator` — 尽量少改

**建议：保留** `__init__` / `init` / `init_ros` / 各 **callback** / **`get_frame`** / **`puppet_arm_publish`** / **`puppet_arm_publish_continuous`** 等与 **ROS I/O** 直接相关的实现。

**仅需核对：**

- 三路 `Image` 的 topic 与真机驱动一致。
- **`get_frame()`** 返回的 **`img_front, img_left, img_right`** 与你在 Step 6 里映射到 **`cam_high / cam_left_wrist / cam_right_wrist`** 的语义一致（与训练标定一致，**不要**接反左右腕）。

---

## Step 4.5：先搭 `run_infer_loop` 骨架（后续 Step 5–9 都填在这里）

先不要追求一次写完，先在 `dz_aloha_infer.py` 建一个最小主循环函数，例如：

```text
def run_infer_loop(args, ros_operator):
  client = ...  # WebSocket client
  rate = rospy.Rate(args.publish_rate)
  while not rospy.is_shutdown():
    result = ros_operator.get_frame()
    if not result: continue
    # Step 5: 图像预处理
    # Step 6: 组 obs
    # Step 7: prompt
    # Step 8: client.infer(obs)
    # Step 9: action -> publish
```

`main()` 里改成：`args -> RosOperator(args) -> run_infer_loop(args, ros_operator)`。  
后面步骤主要就是往这个函数里补内容，不要分散到多个旧 ACT 函数里。

---

## Step 4.6：握手后先做 metadata 校验（避免静默错配）

拿到 `client.get_server_metadata()` 后，建议在进入主循环前校验这些关键项：

- `image_resolution`（与本地 resize 目标一致，默认应对应 176×320）
- `n_external_cameras`、`needs_wrist_camera`（与三路图发送方案一致）
- `needs_session_id`（若为 true，确保你每次 `obs` 带 `session_id`）

若关键字段不一致，直接日志报错并退出，不要“先跑再看”。

---

## Step 5：图像 → server 线协议格式

在 **`get_frame()` 成功返回之后**、调用 **`client.infer(obs)` 之前**，对每路 `uint8` BGR/RGB（按你相机实际）：

1. **`CvBridge.imgmsg_to_cv2`** 得到 **`(H0, W0, 3)`**。
2. **`cv2.resize(..., (width, height), interpolation=cv2.INTER_AREA)`**，其中 **`(width, height) = (320, 176)`**（宽在前、高在后）。
3. 若模型训练为 **RGB**，而 ROS 为 BGR，则 **`cv2.cvtColor(..., cv2.COLOR_BGR2RGB)`**。
4. **`astype(np.uint8)`**，形状 **`(176, 320, 3)`**，放入 obs 的 **图像键**（见 Step 6）。

**不要**在 client 侧做 ImageNet normalize；server / `GrootSimPolicy` 侧按 checkpoint 处理。

---

## Step 6：组装发给 server 的 `obs` 字典

每一控制周期构造 **Python `dict`**。若用 `WebsocketClientPolicy`，`infer()` 会自动加 **`endpoint: "infer"`**；若你自写 client，必须手动加该字段并保持与 server 协议一致。

| Key | 类型 / 形状 | 说明 |
|-----|-------------|------|
| `session_id` | `str` 或 int | 同 Step 0；新任务换新值 |
| `observation/exterior_image_0_left` | `uint8` `(176,320,3)` 或 `(4,176,320,3)` | 固定映射到训练语义 `cam_high` |
| `observation/exterior_image_1_left` | 同上 | 固定映射到训练语义 `cam_left_wrist` |
| `observation/wrist_image_left` | 同上 | 固定映射到训练语义 `cam_right_wrist` |
| `observation/proprio` | `float`，`(14,)` 或 `(1,14)` | 与 `modality.json` 顺序一致；**推荐 `np.asarray(..., dtype=np.float64).reshape(-1)[:14]`** |
| `prompt` | `str` | 使用 **`--task-name`** 或固定文案，例如 `f"Pick the apple: {args.task_name}"` |

**语义映射（固定采用 RoboArena 键名）：**

```text
img_front  → observation/exterior_image_0_left
img_left   → observation/exterior_image_1_left
img_right  → observation/wrist_image_left
```

### Step 6.1：协议参考来源（实现时对照）

- **真机 ROS 收发接口怎么写**：以 `deploy/other_vla_demo_deploy/demo_inference.py` 为准（`RosOperator` / `get_frame` / `puppet_arm_publish`）。
- **client 与 server 交互协议怎么写**：优先对照 `eval_utils/policy_client.py`；如果自写 client，再参考 `test_client_AR.py` 的 send/recv 节奏与字段组织。
- **server 当前真实期望**：以 `socket_test_optimized_AR_aloha.py` 的 `AlohaPolicy._convert_observation` / `image_key_mapping` / `_convert_action` 为最终准绳。

### Step 6.2：`test_client_AR.py` 里“首包 vs 后续包”怎么理解

`test_client_AR.py` 是 **DROID-era 协议测试脚本**，交互骨架直接可用；我们按 agx_aloha 做三处确定改动：

1. **首包与后续包图像时间维不同**  
   - 首包发送单帧（每路 `(H,W,3)`）；  
   - 后续发送多帧（示例是 4 帧 `(4,H,W,3)`）。  
   这和我们 server 侧 `AlohaPolicy` 的时间逻辑一致：新 session 首次 `infer` 用 `T=1`，后续使用 `T=4` 语义。

2. **键名沿用 RoboArena，状态/动作改成 agx**  
   - 图像键固定用 `observation/exterior_image_*` / `observation/wrist_image_left`；  
   - 状态改为 `observation/proprio`（14 维）；  
   - 动作目标改为 `(N,14)`（与 `modality.json` 一致，替代旧的 8 维）。

### Step 6.3：发送节奏（基于交互骨架的落地规则）

- 客户端每个控制周期都发送一次完整 `obs`（图像 + proprio + prompt + session_id）。
- 发送策略固定：首次发送单帧 HWC；后续发送 4 帧 THWC（与 `test_client_AR.py` 一致）。

---

## Step 7：`observation/proprio` 从 `JointState` 拼接

从 **`get_frame()`** 得到的 **`puppet_arm_left` / `puppet_arm_right`**（`sensor_msgs/JointState`）读取 **`.position`**。

**与 `modality.json` 对齐的推荐约定（与常见 ALOHA 7+7 一致时）：**

- 左臂 **`position[0:6]`** → `proprio[0:6]`；**`position[6]`**（第 7 个，夹爪）→ **`proprio[6]`**  
- 右臂 **`position[0:6]`** → **`proprio[7:13]`**；**`position[6]`** → **`proprio[13]`**  

若真机 **关节顺序与训练数据集不一致**，必须在 **本文件内用固定置换矩阵 / 索引表** 显式重排，**不能**假定与 demo 相同。

拼好后：**`obs["observation/proprio"] = proprio`**，dtype 建议 **`np.float64`**。

---

## Step 8：连接 server 与 **reset / infer** 生命周期

**启动阶段（`main` 或 `run_infer` 入口）：**

1. **`rospy.init_node(...)`**，构造 **`RosOperator(args)`**。
2. **创建 client（二选一）**：  
   - 现成封装：`client = WebsocketClientPolicy(host=args.server_host, port=args.server_port)`；  
   - 自写轻量 client：按 **`eval_utils/policy_client.py`** 的协议发收（msgpack、`endpoint`、错误处理）。  
   - 不论哪种方式，首包都应读取 server metadata（`PolicyServerConfig`）用于核对分辨率与字段约定。
3. **（推荐）** 每局任务开始前调用 **`client.reset({...})`**：  
   - `reset_info` 里可带 **`session_id`**（若 server 的 `BasePolicy.reset` 会用到）。  
   - 使用 `WebsocketClientPolicy` 时，`reset()` 已自动设置 `endpoint: "reset"`；自写 client 需手动加。  
   - 与 **`AlohaPolicy.reset`** 联动清 buffer（若你 Wrapper 在 reset 里清帧缓存）。

**控制循环内：**

1. **`rate = rospy.Rate(args.publish_rate)`**。
2. **`while not rospy.is_shutdown()`**：  
   - **`result = ros_operator.get_frame()`**；失败则 `rate.sleep()`、`continue`。  
   - 按 Step 5–7 组 **`obs`**。  
   - **`action = client.infer(obs)`**（注意 `infer` 返回的是 **解包后的对象**：可能是 **`np.ndarray`** 或 **`dict`**，视 server 实现而定；**目标**为 **`ndarray` `shape (N, 14)`**）。  
   - 调用 **Step 9** 把 `action` 发到机器人。  
   - **`rate.sleep()`**。

**`session_id` / `reset` 规则（建议写死在代码里）：**

- 新任务开始：生成新 `session_id`；  
- 进入主循环前：调用一次 `client.reset({"session_id": session_id})`（或最少 `client.reset({})`）；  
- 主循环内：每条 `obs` 都带同一个 `session_id`；  
- 任务结束：再 `client.reset({})` 一次做收尾（便于 server 清缓存/落盘）。

---

## Step 8.5：异机部署（真机 client + 集群 server）通信方式

本项目支持 **client 与 server 不同机器**（这是推荐部署方式）：

- **server**：在集群机器上运行 `socket_test_optimized_AR_aloha.py`（加载 checkpoint、执行模型）。
- **client/infer**：在真机本地运行 `dz_aloha_infer.py`（采集 ROS、发请求、收 action、publish）。

两边只通过 WebSocket 协议通信，不要求共享 Python 环境。

### 8.5.1 直连模式（优先）

- 真机可直接访问集群节点网络时：  
  `--server-host <集群IP或DNS> --server-port 8766`
- 集群侧需允许该端口入站（防火墙/安全组/节点监听地址）。

### 8.5.2 SSH 隧道模式（常见于集群仅内网可达）

若集群节点不能被真机直接访问，可在真机建立端口转发：

```bash
ssh -N -L 8766:127.0.0.1:8766 <user>@<cluster-login-or-node>
```

然后客户端仍用本地地址：

```text
--server-host 127.0.0.1 --server-port 8766
```

> 注：若 server 实际跑在登录节点以外的计算节点，按你们集群策略改成跳板/二级转发即可；核心是把“真机本地端口”转到“server 所在机器端口”。

### 8.5.3 通信自检时机

- 连通性检查放到 **最后联调阶段**（见文末附录 C）。
- 本步骤先按代码逻辑实现，不要让网络细节打断 Step 5–9 开发。

---

## Step 9：解析 server 返回的 `action` 并 `publish`

**目标：** `action` **`shape == (N, 14)`**，与 `modality.json` 的 `action` 一致：

| 下标 | 含义 |
|------|------|
| `0:6` | 左臂 6 关节目标 |
| `6` | 左夹爪 |
| `7:13` | 右臂 6 关节 |
| `13` | 右夹爪 |

**与 ROS `JointState`（每臂 7 个 `position`）对齐：**

```python
left = np.zeros(7, dtype=np.float64)
right = np.zeros(7, dtype=np.float64)
row = action[0]  # 或按控制策略取 row k；第一版常用第 0 步
left[:6] = row[0:6]
left[6] = row[6]
right[:6] = row[7:13]
right[6] = row[13]
ros_operator.puppet_arm_publish(left.tolist(), right.tolist())
```

**多步 chunk（`N>1`）**：可与原 ACT 类似——每控制周期推进 `t`，使用 **`action[t % N]`** 或只执行前 **`K` 步再请求下一次 `infer`**；需与 **`publish_rate`**、机器人跟踪能力一起调，**本文不强制一种策略**，但必须在代码里 **写死一种** 并加注释。

若 server 仍返回 **`dict`**（`action.left_joint_pos` 等），先在客户端 **拼成 `(N,14)`** 再走上表。

### Step 9.1：通信异常与重连（建议第一版就加）

- `client.infer(obs)` 抛异常时：日志打印错误并跳过当步，不要直接崩溃。  
- 连接断开时：重建 client -> 重新 metadata 校验 -> 重新 `reset/session_id` -> 继续循环。  
- publish 前做轻量断言：`isinstance(action, np.ndarray)` 且 `action.ndim == 2`，目标最后维 `14`。

---

## Step 10：`main` 与「给 task name → 跑起来」的入口

**建议结构：**

```text
parse_args()
  → rospy.init_node
  → RosOperator(args)
  → WebsocketClientPolicy(server_host, server_port)
  → （可选）client.reset({...})
  → dreamzero_infer_loop(args, ros_operator, client)
```

**`--task-name`**：在组 `obs` 时 **`obs["prompt"] = args.task_name`**（或前缀 + task_name），即 **自然语言指令**。

**阻塞式「按任意键开始」**（可选，对齐 demo）：在进循环前 **`input("Enter to start infer ...")`**，避免一启动就猛发 `infer`。

---

## Step 11：自检清单（client 侧）

1. 打印 **`obs`** 里三路图像的 **`shape` 与 `dtype`**：应为 **`(176,320,3)` `uint8`**。  
2. 打印 **`obs["observation/proprio"].shape`**：应为 **`(14,)`**。  
3. 打印 **`action.shape`**：目标 **`(N, 14)`**；`N` 与 checkpoint **`action_horizon`** 一致。  
4. 用 **`rostopic echo`** 确认 **`puppet_arm_*_cmd`** 有输出且数值合理。

---

## 与 `dz_aloha_infer.py` 文件的对应关系（便于你拆 diff）

| 原文件大致区域 | 处理方式 |
|----------------|----------|
| L15–173 `get_model_config` / `make_policy` / `get_image` / depth | **删除**或不再 import |
| L196–265 `inference_process` | **删除**；逻辑被 **WebSocket `infer`** 替代 |
| L268–380 `model_inference` | **整体替换** 为 Step 8–9 的循环 |
| L382+ `RosOperator` | **保留**，仅按需改 topic / Step 7 拼接 |
| L681+ `get_arguments` | **增加** server host/port、task/prompt、session |
| L767+ `main` | **改为** Step 10 流程 |

---

## 相关文件

| 文件 | 作用 |
|------|------|
| `deploy/dz_aloha_infer.py` | 真机 ROS 客户端（按本文修改） |
| `eval_utils/policy_client.py` | `WebsocketClientPolicy` 参考实现 |
| `eval_utils/policy_server.py` | server 协议（首包 metadata、`infer`/`reset`） |
| `socket_test_optimized_AR_aloha.py` | `AlohaPolicy` 观测转换与（待 Step 8）动作回包形状 |
| `deploy/deploy_server.md` | server / Wrapper 侧步骤与键名约定 |
| `cobot_data/cobot_dataset/meta/modality.json` | **14 维 state/action 顺序真值** |

---

## 附录 B：`WebsocketClientPolicy` 依赖清单（真机环境）

`eval_utils/policy_client.py` 直接依赖：

- `websockets`（使用 `websockets.sync.client`）
- `typing_extensions`
- `openpi_client`（其中提供 `base_policy` 与 `msgpack_numpy`）

其中 `openpi_client.msgpack_numpy` 会间接需要 msgpack/numpy 相关能力。  
建议在真机环境先做一次导入自检（只要能过，后续再接 ROS）：

```bash
python - <<'PY'
mods = [
  'websockets',
  'typing_extensions',
  'openpi_client',
  'openpi_client.msgpack_numpy',
  'openpi_client.base_policy',
]
for m in mods:
  try:
    __import__(m)
    print('OK ', m)
  except Exception as e:
    print('ERR', m, e)
PY
```

若你们最终自写 client，可不依赖 `openpi_client.BasePolicy`，但仍要保证与 server 的 msgpack/WebSocket 协议一致。

---

## 附录 C：异机通信/SSH 隧道最终联调检查（分步）

在 Step 9 代码完成后再做；**建议先只做本附录，再跑 `dz_aloha_infer.py` 全链路**。默认 server 端口 **`8766`**。

### 0. 先记下三个角色（填你自己的值）

| 角色 | 含义 | 示例（按你们环境替换） |
|------|------|------------------------|
| **计算结点** | `srun` 后跑 `socket_test_optimized_AR_aloha.py` 的那台 | 提示符里如 `ACD1-29` |
| **登录/管理机** | 真机 `ssh` 先进去的那台 | 如 `10.120.48.26`（或 SSH config 里的 `Host` 别名） |
| **真机** | 跑 ROS 客户端或本附录里 `nc`/探活脚本的机器 | 如 `agilex` |

在**计算结点**上确认 hostname / 内网 IP（隧道里 `-L` 的目标有时要用 IP）：

```bash
hostname
hostname -I
```

---

### 1. 在计算结点上：启动 server 并确认已监听

1. 进入仓库、激活环境，按 `general_command.txt` 或你们既定命令启动 server（**保持该终端不关、不按 Ctrl+C**）。
2. 日志里应出现类似：`server listening on 0.0.0.0:8766`。
3. **另开一个 SSH 到同一计算结点的终端**，执行：

```bash
ss -ltnp | awk 'NR==1 || /:8766/'
```

能看到 `0.0.0.0:8766`（或等价监听）再继续下一步。**若此处没有监听，真机侧怎么测都是 `Connection refused`。**

---

### 2. 选拓扑：真机能否直连计算结点 `:8766`

在**真机**上试（把 `<compute>` 换成 hostname 或 `hostname -I` 里与真机互通的 IP）：

```bash
nc -vz <compute> 8766
```

- **成功（open / succeeded）**：可走**直连**，client 用 `--server-host <compute> --server-port 8766`。
- **超时 / refused**：多数集群不允许真机直连计算结点端口，走**下面第 3 步 SSH 隧道**。

---

### 3. 隧道拓扑：真机 → 登录机 → 计算结点 `:8766`

在**真机**上建立本地转发（**占用真机本机 8766**，与 server 端口数字一致即可）：

```bash
ssh -N -L 8766:<compute>:8766 <user>@<login>
```

示例（仅作格式参考，请替换为你的计算结点名与账号）：

```bash
ssh -N -L 8766:ACD1-29:8766 haoangli@10.120.48.26
```

- **`<login>`**：必须与你平时能 `ssh` 登录集群的写法一致（IP、`Host` 别名均可）。
- **`<compute>`**：填计算结点 **hostname**；若登录机解析不到或转发失败，可改为该计算结点的**内网 IP**（与在计算结点上 `hostname -I` 对照）。
- **认证**：若出现 `Permission denied`，说明卡在 **「真机 → 登录机」SSH 认证**，尚未连到计算结点。处理要点：
  - 与「不带 `-N -L`、能成功登录」的那条命令使用**同一用户、同一密钥**；若平时用密钥，需加 `-i /path/to/key`，或在 `~/.ssh/config` 里为 `<login>` 配好 `IdentityFile` 后用 `Host` 别名执行隧道。
  - 可用 `ssh -v <user>@<login>` 看最终走的是 `publickey` 还是 `password`（集群若仅允许公钥，则不能只靠密码）。
- 该 `ssh -N -L ...` 终端会**一直挂起**，属正常；**另开新终端**做第 4、5 步。

**若登录机无法 TCP 连到计算结点 `8766`**（防火墙/路由策略）：需在集群内按 Step 8.5.2 改用**反向隧道**或其它运维允许的通路；本附录不展开。

---

### 4. 在真机上：端口连通（隧道模式用本机回环）

- **已建隧道**（第 3 步）时：

```bash
nc -vz 127.0.0.1 8766
```

- **直连模式**（第 2 步 `nc` 已成功）时，可再测一次：

```bash
nc -vz <compute> 8766
```

`Connection refused`：server 未起、端口不对、或隧道目标 `<compute>` 写错。  
`Permission denied`：**不是**端口问题，见第 3 步 SSH 说明。

---

### 5. 在真机上：协议层探活（metadata，仍不跑 infer）

在 **dreamzero 仓库根**（或设置好 `PYTHONPATH` 指向仓库根）下执行；**隧道模式**下 `host` 用 `127.0.0.1`，**直连模式**下用 `<compute>`。

```bash
cd /path/to/dreamzero
python - <<'PY'
from eval_utils.policy_client import WebsocketClientPolicy
host = "127.0.0.1"  # 隧道模式；直连则改为计算结点 IP/hostname
port = 8766
c = WebsocketClientPolicy(host=host, port=port)
print(c.get_server_metadata())
PY
```

能打印出字典（含 `image_resolution`、`n_external_cameras`、`needs_wrist_camera` 等）即 **WebSocket + 首包 metadata 正常**。再进入 Step 4.6 与 `dz_aloha_infer.py` 全链路。

---

### 6. 推荐联调顺序（小结）

1. 计算结点：起 server → `ss` 确认 `8766` 监听。  
2. 真机：`nc` 判直连或隧道。  
3. 隧道：真机 `ssh -N -L ...`（认证与日常登录一致）→ `nc -vz 127.0.0.1 8766`。  
4. 真机：`get_server_metadata()` 探活。  
5. 最后：`dz_aloha_infer.py`（`--server-host` / `--server-port` 与上面选择一致）。

---

## 附录 A：审阅补遗（实现 `dz_aloha_infer.py` 时易错点）

以下为对照 **`eval_utils/policy_client.py`**、**`eval_utils/policy_server.py`**、**`AlohaPolicy.reset` / `infer`** 后的核对结论；**正文 Step 0–11 仍成立**，此处只补「容易写错」的细节。

1. **`WebsocketClientPolicy.infer` 的返回值类型**  
   源码里类型注解写 **`-> Dict`**，但 server 在 `infer` 分支里是 **`packer.pack(action)`**，而 **`AlohaPolicy.infer` 当前返回 `numpy.ndarray`**。因此客户端 **`action = client.infer(obs)` 得到的多半是 `np.ndarray`，不是 dict**。Step 9 已写「可能是 ndarray 或 dict」；实现时 **应先 `isinstance(action, dict)` 再分支**，不要假定一定是 dict。

2. **`reset` 的回包格式与 `infer` 不同**  
   `policy_server` 在 **`endpoint == "reset"`** 时发送的是 **纯 Python 字符串** **`"reset successful"`**（**未经** `msgpack_numpy.pack`）。`WebsocketClientPolicy.reset` 里 **`recv()` 直接返回该 str**。这与 **`infer` 收到 bytes 再 `unpackb`** 不同；**不要在 reset 之后对返回值做 `unpackb`**。

3. **错误路径**  
   server 在异常时可能 **`await websocket.send(traceback.format_exc())`**，即 **字符串** traceback。`WebsocketClientPolicy.infer` 对 **`isinstance(response, str)`** 会 **`raise RuntimeError`**。客户端应 **捕获并打日志**，便于区分「机器人 / ROS 问题」与「server 崩了」。

4. **`PYTHONPATH` / 工作目录**  
   `from eval_utils.policy_client import ...` 要求 **在仓库根 `dreamzero` 下运行**，或设置 **`PYTHONPATH=/path/to/dreamzero`**（按你部署方式二选一）。正文 Step 1 未强调，此处补上。

5. **`session_id` 与 `client.reset` 的关系**  
   - **`AlohaPolicy.infer`**：若 **`obs["session_id"]` 与上次不同**，会 **`_reset_state()`** 并更新内部 session。  
   - **`AlohaPolicy.reset(reset_info)`**（WebSocket **`endpoint=="reset"`**）：只调 **`_reset_state()`**，**不**改 `_current_session_id`；但会 **`_is_first_call = True`**，等价于 **新 episode 的视频时间维又从 T=1 开始**。  
   实践上：**新开一局**可 **二选一**——要么 **`client.reset({})` 后再 infer`**（`session_id` 可不变），要么 **不换 reset、只换 `obs["session_id"]`** 触发 server 侧清 buffer。**不要**两种都不做却期望 buffer 清空。

6. **阻塞时间与 WebSocket ping**  
   DreamZero 单次 **`infer` 可能极慢**；`policy_client` 已把 **`ping_timeout` 调到 600s**。若仍断连，可再加大或暂时关心跳（需改 client 连接参数），否则长推理中途会被库断开。

7. **`cv2` import**  
   Step 5 写了 **`cv2.resize`**，若删掉原 demo 里未显式 import 的路径，需在 **`dz_aloha_infer.py` 顶部增加 `import cv2`**。

---

*若你之后把 `_convert_action` 固定为 `(N,14)` 并改 `infer` 的 docstring，可在本文 Step 9 删除「兼容 dict / (N,8)」的说明，只保留 `(N,14)`。*
