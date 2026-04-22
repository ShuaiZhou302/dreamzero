# Deploy：真机（Cobot）与集群 DreamZero Server 联调

本目录说明如何在 **Aloha / Cobot 真机** 上准备**最小 Python 环境**（不等同于仓库根目录 [`README.md`](../README.md) 的完整依赖），用于：

- 跑 [`test_client_AR.py`](../test_client_AR.py) 做 WebSocket 探活 / 联调；
- 跑 [`deploy/dz_aloha_infer.py`](dz_aloha_infer.py) 做 ROS + DreamZero 推理客户端。

**Server（模型、GPU）仍在集群计算结点**；真机只需能连上 `WebsocketClientPolicy` 与可选 ROS。

---

## 1. 在真机上克隆仓库

将下面路径换成你在 Cobot 上的目录（示例：`~/workspace/dreamzero`）。

```bash
mkdir -p ~/workspace && cd ~/workspace
git clone <你的 dreamzero 仓库 URL> dreamzero
cd dreamzero
```

若使用 SSH：

```bash
git clone git@github.com:<org>/dreamzero.git ~/workspace/dreamzero
cd ~/workspace/dreamzero
```

---

## 2. 创建 Conda 环境（仅客户端依赖）

**Python 版本：3.11**（与仓库 `pyproject.toml` 的 `requires-python` 一致）。

```bash
conda create -n dz_cobot_client python=3.11 -y
conda activate dz_cobot_client
```

安装**最小 pip 依赖**（来自本目录 `requirements-cobot-client.txt`，对齐 [`deploy_infer.md`](deploy_infer.md) 附录 B + 仓库中 `openpi-client` 版本）：

```bash
cd ~/workspace/dreamzero
pip install -U pip
pip install -r deploy/requirements-cobot-client.txt
```

---

## 3. 依赖自检（附录 B：导入一遍）

在仓库根目录执行（**必须**能 `import eval_utils`，故需设置 `PYTHONPATH`）：

```bash
cd ~/workspace/dreamzero
export PYTHONPATH="${PWD}"
python - <<'PY'
mods = [
    "websockets",
    "typing_extensions",
    "openpi_client",
    "openpi_client.msgpack_numpy",
    "openpi_client.base_policy",
    "numpy",
    "cv2",
]
for m in mods:
    try:
        __import__(m)
        print("OK ", m)
    except Exception as e:
        print("ERR", m, e)
PY
```

全部 `OK` 再继续。若 `ERR`，按报错补装或检查是否在仓库根且已 `export PYTHONPATH`。

---

## 4. SSH 隧道（真机无法直连计算结点时）

在 **Cobot 真机** 开一个终端，**保持不关**（把 `10.120.48.106` 换成你当前计算结点 IP，`haoangli@10.120.48.26` 换成你的登录方式）：

```bash
ssh -N -L 8766:10.120.48.106:8766 haoangli@10.120.48.26
```

另开终端测端口：

```bash
nc -vz 127.0.0.1 8766
```

期望：`Connection to 127.0.0.1 8766 port [tcp/*] succeeded!`  
更完整分步见 [`deploy_infer.md`](deploy_infer.md) **附录 C**。

---

## 5. 跑 `test_client_AR.py`（无 ROS）

**说明**：当前脚本会 `import eval_utils.policy_server`（仅用到 `PolicyServerConfig`），依赖与附录 B 一致，**不需要**根目录 README 里的 torch / 训练栈。

```bash
cd ~/workspace/dreamzero
conda activate dz_cobot_client
export PYTHONPATH="${PWD}"

# 隧道模式下 host 用 127.0.0.1；若真机能直连计算结点则改为计算结点 IP
python test_client_AR.py --host 127.0.0.1 --port 8766
```

若仅用零图像、不读 `debug_image/`：

```bash
python test_client_AR.py --host 127.0.0.1 --port 8766 --use-zero-images
```

---

## 6. 跑 `dz_aloha_infer.py`（需要 ROS）

除第 2 节的 pip 包外，**还需要本机已配置 ROS**（与你们 `demo_inference` 一致），例如：

- `rospy`
- `sensor_msgs`、`geometry_msgs`、`nav_msgs`、`std_msgs`
- `cv_bridge`

**不在** `requirements-cobot-client.txt` 内，通常通过系统 ROS 或你们现有 Catkin 环境提供。

示例（ROS Noetic，按你机器实际修改 `setup.bash` 路径）：

```bash
source /opt/ros/noetic/setup.bash
# 若有工作空间：
# source ~/catkin_ws/devel/setup.bash

cd ~/workspace/dreamzero
conda activate dz_cobot_client
export PYTHONPATH="${PWD}"

python deploy/dz_aloha_infer.py \
  --server-host 127.0.0.1 \
  --server-port 8766 \
  --prompt "你的任务描述"
```

`--server-host` / `--server-port` 与隧道或直连一致；其余相机 / 关节 topic 见 `dz_aloha_infer.py` 内 `get_arguments()` 默认值或按真机改。

---

## 7. 与根目录 README 的关系

| 文档 | 用途 |
|------|------|
| [`README.md`](../README.md) | 完整项目：训练、评测、大量 CUDA / 科学计算依赖 |
| 本 `Readme.md` + `requirements-cobot-client.txt` | **仅**真机侧：`WebsocketClientPolicy` + `test_client_AR` / `dz_aloha_infer` 所需最小 pip 集 |

真机**不必**按根目录 README 安装 torch、transformers、deepspeed 等。

---

## 8. 附录：集群侧登录与 Server 端口检查（简要）

登录管理 / 登录机（示例）：

```bash
ssh haoangli@10.120.48.26
```

在计算结点启动 server 后，在**该计算结点**上确认监听（端口以你启动参数为准，默认 `8766`）：

```bash
hostname
hostname -I
ss -ltnp | awk 'NR==1 || /:8766/'
```

将 `hostname -I` 中与集群内网一致的 IP 填入第 4 步隧道命令中的 `<计算结点IP>`。

---

若附录 B 自检或 `get_server_metadata()` 失败，把完整报错贴出排查即可。
