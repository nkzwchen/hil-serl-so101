# 阶段 A：让 `ros2 launch so101_bringup ...` 跑起来（ROS2 装进 hil-serl）

> 目标命令：
> ```bash
> ros2 launch so101_bringup so101_teleoperate.launch.py teleop_mode:=real expert:=human
> ```
> 环境事实：WSL2 / Ubuntu 24.04.2 / 原无 ROS2
> **方案：RoboStack（conda 版 ROS2 Humble）装进现有的 `hil-serl` 环境**
> 端口以 `lerobot/useful_script/control.sh` 为准

---

## 1. 已完成的代码与配置修复（无需你操作）

### 1.1 lerobot 导入路径（`so101_ros2` ↔ 官方 lerobot 0.5.1）

唯一的不兼容点是导入路径。已改（备份在 `/tmp/bridge.py.bak`、`/tmp/build.py.bak`）：

| 文件 | 改动 |
|---|---|
| `so101_ros2_bridge/so101_ros2_bridge/bridge/build.py:29-30` | `robots.so101_follower` → `robots.so_follower`；`teleoperators.so101_leader` → `teleoperators.so_leader` |
| `so101_ros2_bridge/so101_ros2_bridge/bridge/bridge.py:46-47` | 同上 |

实测 4 个符号导入成功：`SO101Follower→SOFollower`、`SO101FollowerConfig→SOFollowerRobotConfig`、`SO101Leader→SOLeader`、`SO101LeaderConfig→SOLeaderTeleopConfig`。
`SO101FollowerConfig` 的字段 `['port','disable_torque_on_disconnect','max_relative_target','cameras','use_degrees','id','calibration_dir']` 与 `build.py` 传参完全一致。

### 1.2 两个会导致"机械臂不动/接错设备"的配置错误

**错误 A：端口冲突。** 原 `so101_follower_params.yaml` 和 `so101_leader_params.yaml` **都写 `/dev/ttyACM0`** —— 两个节点会抢同一个串口。已按 `control.sh` 改正：

| 设备 | control.sh | 修正后 yaml |
|---|---|---|
| follower | `/dev/ttyACM1` | `/dev/ttyACM1` ✅ |
| leader | `/dev/ttyACM0` | `/dev/ttyACM0` ✅ |

**错误 B：标定文件 id / 目录不匹配。** 原 leader 用 `id: "Gili"`，读的是仓库自带的 `config/calibration/Gili.json`（**跟你的臂不是同一台的标定数据**）。而 `control.sh` 用的是另一套：

```
/home/nkzwchen/.cache/huggingface/calibration/robots/so_follower/so101_follower_main.json      ← follower
/home/nkzwchen/.cache/huggingface/calibration/teleoperators/so_leader/so101_leader_main.json   ← leader
```

已改为使用 `control.sh` 那套，并显式给出 `calibration_dir`。路径拼接逻辑（`lerobot/robots/robot.py:53` + `teleoperators/teleoperator.py:53`）：

```python
calibration_fpath = calibration_dir / f"{id}.json"
# 且若不指定 calibration_dir，默认是 HF_LEROBOT_HOME/lerobot/calibration/{robots|teleoperators}/{name}
# 实测默认解析到 /root/.cache/... → 与你的文件位置不符，所以必须显式指定
```

**错误 C：参数文件的顶层键与真实节点名不符（这个最隐蔽）。**

`so101_ros2_bridge.launch.py` 用 `namespace=<type>` + `name=so101_<type>_interface`，所以真实节点全限定名是：

```
/follower/so101_follower_interface
/leader/so101_leader_interface
```

而两个 yaml 的顶层键写的是 `so101_follower_ros2_bridge` / `so101_leader_ros2_bridge` —— **不匹配，参数会被 ROS2 静默忽略**（节点就用代码里的默认值：leader 默认端口恰好也是 `/dev/ttyACM0`，所以端口"看起来对"，但 `id` 会退化成 `Tzili`/`Gili`）。已改为同时提供 `/<type>/so101_<type>_interface` 与 `/**` 两种键，装好后用下面的命令确认哪一种生效：

```bash
ros2 param list /follower/so101_follower_interface
```

> 另外 `LeaderBridge.read_parameters()` **没有声明** `disable_torque_on_disconnect`，所以 leader yaml 里刻意不写这一项，避免 "parameter not declared"。

---

## 2. 环境选型结论

**用 RoboStack 装进 `hil-serl` 环境**，理由：

| 项 | 值 |
|---|---|
| 系统 | Ubuntu 24.04 → 装不了 Humble 的 deb（Humble 目标平台是 22.04） |
| 目标环境 | `hil-serl`，**Python 3.10.21** |
| RoboStack 是否有 py310 构建 | ✅ 有。实查：`ros-humble-ros2-control 2.22.0 py310h7c61026_3`（robostack-staging）、`ros-humble-rclpy 3.3.7 py310h7c61026_3` |
| 结论 | 系统保持 24.04 不动，ROS2 完全在 conda 里 |

### 已用 `--dry-run` 验证依赖计划（实测，284 个包）

```
无 REMOVED，无 conflict
DOWNGRADED: icu 78.3 → 75.1      （不影响任何 Python 侧功能）
SUPERSEDED: setuptools 84.0.0 → 61.0.0（conda-forge 通道优先级导致）
REVISED:    libsqlite
```

⚠️ **要提前知道的两件事**：

1. **会新装 conda 版 `numpy 1.26.4`**。你现在 `hil-serl` 里是 pip 装的 `numpy 2.2.6`。conda 版会覆盖/遮蔽它。
   → 这对阶段 A **无害**，但**阶段 B 装 jax 时必须一起考虑**（jax 0.4.x 配 numpy 1.26 是安全的；不要事后单独 `pip install -U numpy`，否则又破坏 ROS2 的 numpy 依赖）。
2. **`setuptools` 会退回 61.0.0**。如果后续要用新版 setuptools 的 PEP 660 editable install，需要留意。

---

## 3. 安装 ROS2 Humble（装进 hil-serl）—— ✅ 已完成

**已执行并验证**（2026-09-26）：

```bash
conda config --env --add channels conda-forge
conda config --env --add channels robostack-staging
conda config --env --add channels robostack-humble

conda install -y \
  ros-humble-ros2-control ros-humble-ros2-controllers ros-humble-hardware-interface \
  ros-humble-joint-trajectory-controller ros-humble-gripper-controllers
```

**结果**：conda 包数 27 → **311**，其中 **161 个 ros-humble 包**；`ros2` CLI、`rclpy` 均已就位；`numpy` 按预期变为 conda 版 **1.26.4**（原 pip 2.2.6）。

装了哪些（关键项）：`controller_manager 2.22.0`、`rclpy 3.3.7`、`hardware_interface`、`joint_trajectory_controller`、`gripper_controllers`、`joint_state_broadcaster`。

### ⚠️ 装完后的 4 个坑（→ 见 §3.5 与 `fix_hilserl_ros2.sh`）

装完**不能直接用**，暴露出 4 个问题，已定位并给出修复脚本：

| # | 问题 | 现象 | 根因 |
|---|---|---|---|
| 1 | **`typing` backport 遮蔽 stdlib** | `ros2` 任何子命令都崩溃 | `site-packages/typing.py`（pip 装的 `typing 3.7.4.3`）抢在 stdlib 前被 import，其 `_abc_registry` 在 Python 3.10 已被移除 |
| 2 | `ROS_DISTRO` 为空 | `echo $ROS_DISTRO` 输出空行 | RoboStack 的 `ros-workspace` 包 activate 脚本只导出 `AMENT_PREFIX_PATH`，未导出 `ROS_DISTRO`（不是安装失败） |
| 3 | 通道顺序错误 | 补装包报 `LibMambaUnsatisfiableError` | `.condarc` 里 `robostack-humble` 在 `robostack-staging` 之前，求解器选中新版 `ros2-distro-mutex`，与环境已装的 `0.3.0` 冲突 |
| 4 | `xacro` / `robot_state_publisher` 缺失 | 后续 `rsp.launch.py` 会失败 | 首次只装了核心包，且直接补装会被问题 3 挡住 |

### 3.5 修复（跑这个脚本）

```bash
bash /home/nkzwchen/pi_project/fix_hilserl_ros2.sh
```

脚本做四件事：
1. `pip uninstall -y typing`（stdlib 自带 `typing`，卸载 PyPI backport 无副作用）
2. 往 `activate.d` 追加 `ROS_DISTRO=humble` / `ROS_VERSION=2` / `ROS_PYTHON_VERSION=3`
3. 改写 `$CONDA_PREFIX/.condarc`，把 `robostack-staging` 提到最前、去掉 `robostack-humble`
4. 补装缺失依赖，**钉到与现有 `ros2-distro-mutex 0.3.0` 匹配的构建**：

```bash
conda install -y -p $CONDA_PREFIX \
  "ros-humble-xacro=2.0.8=py310h7c61026_3" \
  "ros-humble-robot-state-publisher=3.0.2=py310h7c61026_3" \
  ros-humble-eigen3-cmake-module ros-humble-moveit-common ros-humble-orocos-kdl-vendor
```

> 已 `--dry-run` 验证：**新增 10 个包，零降级、零移除、无冲突**（含 `kdl-parser` / `urdf` / `urdf-parser-plugin` / `urdfdom`(+headers) / `tinyxml-vendor` / `tinyxml`）。
> 证据文件：`.dryrun4.txt`

### 3.6 修复后验证

```bash
conda activate hil-serl
python --version                       # 3.10.21
echo $ROS_DISTRO                       # humble（重新 activate 后）
ros2 pkg list | wc -l                  # 预期 160+（修复前崩溃/只有 12）
for p in robot_state_publisher xacro controller_manager \
         joint_trajectory_controller gripper_controllers \
         joint_state_broadcaster hardware_interface; do
  ros2 pkg list | grep -qx "$p" && echo "✅ $p" || echo "❌ $p"
done
xacro --version
```

### 3.7 ⚠️ 已知限制：`usb_cam` 在本环境无法安装

实查 `ros-humble-usb-cam` 的 linux-64 构建只有 4 个：

```
0.8.1  np126py311hbc2a38a_7 / _13      ← py311
0.8.1  np2py312h2ed9cc7_15 / _18       ← py312
```

**没有 py310 构建**，所以在 Python 3.10 的 hil-serl 环境里装不上。
含义：`so101_bringup` 的 `camera.launch.py` 在阶段 A 起不来相机节点。

**处理策略（分两阶段）**：

| 阶段 | 是否需要相机 | 做法 |
|---|---|---|
| **阶段 A**（先跑通 `expert:=human` 遥操作） | 不需要 | 保持现状，相机节点自然缺失，不影响 Leader→Follower 跟随 |
| **阶段 B**（HIL-SERL 观测 / reward classifier） | 需要 | 二选一：<br>① 用**系统 Python**（3.12，有 `cv2`）写一个自己的相机节点发布 `sensor_msgs/Image` 到 `/follower/cam_*/image_raw`，发布端与 Python 版本无关；<br>② 在**另一个 conda env（3.11/3.12）** 装 `ros-humble-usb-cam` 跑相机节点，用 `ROS_DOMAIN_ID` 通信 |

> 备选：系统 apt 装 `ros-jazzy-usb-cam` 也能跑（24.04 原生），用 `ROS_DOMAIN_ID` 跨发行版通信——但跨发行版 DDS 兼容性需实测，不作为首选。

### 3.8 长期建议

当前环境是 RoboStack 的**较旧快照**（`ros2-distro-mutex 0.3.0`、`controller_manager 2.22.0`、`rclpy 3.3.7`），所以只能配到 2023 年左右的包版本（这也正是 `usb_cam` 装不上的原因）。
如果后续要长期维护，建议**用 `ros-humble-ros2-control` 2.54.0（`np2py312`）+ Python 3.12 重建一份环境**，版本一致性更好、可选包更全。代价是 `lerobot` 与 ROS2 的 Python 版本要靠 `LECONDA_SITE_PACKAGES` 或同版本安装来打通。

> 如果 `conda install` 报 `EnvironmentNotWritableError`：`hil-serl` 目录属主是 root，需要 `sudo` 或先 `sudo chown -R $USER:$USER /home/nkzwchen/miniforge3/envs/hil-serl`。

---

## 4. 让 ROS2 找到 lerobot

`so101_ros2_bridge/utils/core.py:26` 会读 `LECONDA_SITE_PACKAGES` 并插入 `sys.path`：

```bash
export LECONDA_SITE_PACKAGES=/home/nkzwchen/miniforge3/envs/lerobot/lib/python3.12/site-packages
```

> 注意版本差：ROS2 在 3.10，lerobot 在 3.12。`lerobot` 主体是纯 Python，跨小版本 import 一般可行。
> **如果出现 `.so` 加载失败**（例如 torch/torchcodec 相关），改成在 hil-serl 里装一份 lerobot：
> ```bash
> pip install -e /home/nkzwchen/pi_project/lerobot
> ```

验证：

```bash
python -c "
import sys; sys.path.insert(0, '$LECONDA_SITE_PACKAGES')
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
print('lerobot OK')
"
```

---

## 5. 编译 so101_ros2 工作区

### ⚠️ 不要照抄 `so101_ros2/build.sh`

它有 3 处在 conda 环境下**必然失败**：

| 行 | 问题 | 处理 |
|---|---|---|
| `source /opt/ros/humble/setup.bash` | 路径不存在 | 改成 `conda activate hil-serl` |
| `sudo rosdep init` / `rosdep update` / `rosdep install --from-paths src` | rosdep 不认识 conda 包，会尝试 apt 安装并失败 | **整段跳过**（依赖已由 conda 装齐） |
| `sudo apt install python3-rosdep` | 不需要 | 跳过 |

手动执行：

```bash
conda activate hil-serl
export LECONDA_SITE_PACKAGES=/home/nkzwchen/miniforge3/envs/lerobot/lib/python3.12/site-packages

mkdir -p /home/nkzwchen/pi_project/ros2_ws/src
ln -sfn /home/nkzwchen/pi_project/so101_ros2/{so101_bringup,so101_controller,so101_description,so101_hardware_interface,so101_ros2_bridge,so101_teleop} \
        /home/nkzwchen/pi_project/ros2_ws/src/

cd /home/nkzwchen/pi_project/ros2_ws
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/local_setup.bash
```

### 可能踩的坑

| 症状 | 处理 |
|---|---|
| `find_package(moveit_common REQUIRED)` 失败 | 已在 §3.4 装 |
| `find_package(eigen3_cmake_module)` 失败 | 已在 §3.4 装 |
| 链接到系统 `/usr/lib` 而非 conda 库 | 编译前 `unset LD_LIBRARY_PATH`，然后重新 `conda activate hil-serl`；确认 `CMAKE_PREFIX_PATH` 含 `$CONDA_PREFIX` |
| `so101_hardware_bridge` 插件加载失败 | 确认 `$CONDA_PREFIX/lib` 在 `LD_LIBRARY_PATH` 和 `AMENT_PREFIX_PATH` 中 |
| `colcon build` 找不到 python | 确认用的是 conda 的 python：`which python` 应指向 `hil-serl/bin/python` |

---

## 6. 设备直通（usbipd）

Windows 侧已确认装有 `usbipd-win`。**Windows PowerShell（管理员）**：

```powershell
usbipd list                              # 找串口(CH340/CP210x/ACM)与摄像头
usbipd bind   --busid <BUSID>            # 首次，一次即可
usbipd attach --wsl --busid <BUSID>      # 每次插拔后都要执行
```

**WSL 侧验证**（同时确认端口与 control.sh 一致）：

```bash
ls -la /dev/ttyACM* /dev/ttyUSB* /dev/video*
```

预期看到 `/dev/ttyACM0`（leader）和 `/dev/ttyACM1`（follower）。
**如果顺序反了**（取决于插拔顺序），只需交换两个 yaml 里的 `port` 值，或按 §1.2 方式改。

注意 `record.sh` 里写的是 `/dev/ttyUSB0|1`，与 `control.sh` 不一致 —— **按你的要求，一律以 `control.sh` 为准**。

---

## 7. 启动目标命令

```bash
source /home/nkzwchen/miniforge3/etc/profile.d/conda.sh
conda activate hil-serl

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export LECONDA_SITE_PACKAGES=/home/nkzwchen/miniforge3/envs/lerobot/lib/python3.12/site-packages
source /home/nkzwchen/pi_project/ros2_ws/install/local_setup.bash

ros2 launch so101_bringup so101_teleoperate.launch.py \
  teleop_mode:=real \
  expert:=human
```

`so101_bringup/launch/so101_teleoperate.launch.py` 的启动编排：

| 顺序 | 内容 | 延时 |
|---|---|---|
| 1 | follower 的 robot_state_publisher | — |
| 2 | leader（`expert:=human` 时才起） | — |
| 3 | follower（机器人节点 + controller_manager + spawner） | 5s |
| 4 | cameras | — |
| 5 | leader_teleop 组件（C++） | 15s |
| 6 | RViz（仅 `display:=true`） | 5s |

---

## 8. 验收清单（阶段 A 通过标准）

```bash
# ① 节点名正确（验证 §1.2 错误 C 的修复）
ros2 node list
#   期望：/follower/so101_follower_interface, /leader/so101_leader_interface

# ② 参数确实生效（关键！）
ros2 param get /follower/so101_follower_interface port          # /dev/ttyACM1
ros2 param get /leader/so101_leader_interface   port            # /dev/ttyACM0
ros2 param get /leader/so101_leader_interface   id              # so101_leader_main
ros2 param get /follower/so101_follower_interface calibration_dir
#   若报参数不存在 → §1.2 的顶层键写法没匹配上，需要按 ros2 param list 的实际名字改

# ③ 话题
ros2 topic list
#   期望：/follower/joint_states, /follower/joint_states_raw,
#         /follower/cam_main/image_raw, /follower/cam_wrist/image_raw, /leader/joint_states

# ④ 频率
ros2 topic hz /follower/joint_states_raw     # ~50 Hz (publish_rate=50)
ros2 topic hz /follower/cam_main/image_raw   # ~30 Hz

# ⑤ 控制器
ros2 control list_controllers
#   期望：joint_state_broadcaster[active], arm_controller[active], gripper_controller[active]

# ⑥ 最终验收：手动移动 Leader 臂 → Follower 跟随
```

---

## 9. 进度：阶段 A ✅ 已完成

| # | 事项 | 状态 |
|---|---|---|
| 1 | lerobot 导入路径修复（`build.py` / `bridge.py`） | ✅ |
| 2 | 端口 / 标定 / 参数键三处配置修复 | ✅ |
| 3 | conda 装 RoboStack ROS2（py312 重建） | ✅ 490 conda 包 / 250 个 share 包 |
| 4 | lerobot 原生安装（含 `[feetech]` extra） | ✅ 0.5.1 + torch 2.10.0+cu128 |
| 5 | 编译 so101_ros2 | ✅ 6 包全过 |
| 6 | usbipd 直通（两路 CH343） | ✅ `/dev/ttyACM0` leader + `/dev/ttyACM1` follower |
| 7 | **遥操作实测** | ✅ **Leader 臂能遥控 Follower 臂** |

### 9.1 阶段 A 踩的 8 个坑（按发现顺序）

| # | 现象 | 根因 | 修复 |
|---|---|---|---|
| 1 | `sensor_msgs` 等导入失败 | `so101_ros2` 用旧 fork 的 lerobot 导入路径 | `robots.so101_follower` → `robots.so_follower`（4 行） |
| 2 | 两个桥接抢同一串口 | 两个 yaml 都写 `/dev/ttyACM0` | follower → `/dev/ttyACM1`（以 `control.sh` 为准） |
| 3 | 标定数据不是本机的 | `id: "Gili"` 读仓库自带标定 | 改用 `control.sh` 的 `so101_leader_main` + 显式 `calibration_dir` |
| 4 | yaml 参数被静默忽略 | 顶层键 `so101_follower_ros2_bridge` ≠ 真实节点名 `/follower/so101_follower_interface` | 改用全限定节点名 + `/**` 双保险 |
| 5 | `pip`/`conda`/`ros2` 全部崩溃 | `PYTHONPATH` 被 ament 钩子污染（路径拼成 `python312`），里面 `typing.py` 是 3.7 backport | 重建 py312 + `unset PYTHONPATH`（`so101_env.sh` 已处理） |
| 6 | `ros2 pkg` / `ros2 launch` 不存在 | 包清单漏了 `ros2pkg` / `ros2launch` / `ros2run` 等 CLI 包 | 补装 11 个 ros2 CLI 包 |
| 7 | 节点创建失败、进程秒退 | `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` 导致 CycloneDDS participant 耗尽（"Failed to find a free participant index for domain 0"） | **改用默认 `rmw_fastrtps_cpp`**（已从 activate.d 删除该变量） |
| 8 | 每帧刷屏 `Failed to send commands to robot: 10` | lerobot 0.5.1 的 `ensure_safe_goal_position` 只接受 `float`/`dict`，`isinstance(10, float)` 为 False → `TypeError(10)`；但 `SOFollowerConfig` 类型标注却是 `int \| None` | yaml 写 **`10.0`**（浮点！）+ 参数声明改 double |

### 9.2 阶段 A 最终验证（实测通过）

```bash
source /home/nkzwchen/pi_project/so101_env.sh
ros2 launch so101_bringup so101_teleoperate.launch.py \
  teleop_mode:=real expert:=human use_cameras:=false
# → 移动 Leader 臂，Follower 跟随 ✅
```

已确认工作的节点链路：

```
Leader 臂 ──串口(ttyACM0)──> leader_ros2_node ──/leader/joint_states──>
  LeaderTeleopComponent ──/follower/arm_controller/joint_trajectory──>
  arm_controller ──> So101HardwareBridge ──/joint_commands──>
  follower_ros2_node ──串口(ttyACM1)──> Follower 舵机
```

`max_relative_target=10.0` 生效中，会把过大的目标钳制到每步 10°，日志里看到
`Relative goal position magnitude had to be clamped to be safe` 属正常保护。

---

## 10. 下一步：阶段 B（RLPD / HIL-SERL）

阶段 A 只解决了"能遥控"。阶段 B 要做算法，还有这些缺口：

| # | 事项 | 说明 |
|---|---|---|
| B1 | **装 `serl_launcher` 的 ML 依赖** | jax / flax / optax / distrax / tensorflow / tfp / chex / wandb / einops / imageio / moviepy 等（旧 3.10 环境被删，这些全没了） |
| B2 | **第三视角相机** | reward classifier 需要；目前只有 `cam_main` / `cam_wrist` 两路，`so101_cameras.yaml` 要加第三路 |
| B3 | **接相机（usb_cam 已装）** | 插上 USB 相机后按实际枚举改 `so101_usb_cam.yaml` 的 `video_device` |
| B4 | **写 `SO101Env`（gym）** | 观测=关节角+图像；动作=关节增量（归一化到 [-1,1]）；`proprio_keys` 不含 `tcp_force`/`tcp_torque` |
| B5 | **写干预 wrapper** | 用 Leader 臂当 `intervene_action` 源（复用现成链路），动作语义要与 policy 严格一致 |
| B6 | **奖励设计** | 建议塑形奖励；`discount` 从 0.97 提到 0.99 |
| B7 | **⚠️ 显存约束** | GPU 是 **RTX 4060 Laptop 8GB**，而 HIL-SERL 文档推荐 24GB（4090）。RLPD 的 `batch_size` / `critic_ensemble_size` / 图像分辨率都要调小 |

### 遗留的小问题（不阻塞）

1. `gripper_controller` 的 spawner 偶尔报 `rmw handle is invalid` —— 第一次启动时的 DDS 时序问题，`joint_state_broadcaster` 和 `arm_controller` 都正常。若影响夹爪控制，可在 launch 里给 spawner 加 `--controller-manager-timeout 30`。
2. `robot_state_publisher` 警告 URDF 的 root link 带 inertia（KDL 限制），无害。
3. `ros2 pkg list` 在当前 WSL2 里很慢/会卡 —— 用 `$CONDA_PREFIX/share/<pkg>/package.xml` 代替。
4. 启动瞬间前 2 帧可能出现 `nan` 目标（被我加的限流日志抓到过一次），随后立即正常。
