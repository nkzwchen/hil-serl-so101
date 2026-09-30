# so101_ros2 通信原理学习笔记

这份笔记整理 `so101_ros2` 如何与 **相机** 和 **SO-101 机械臂** 通信，以及这些数据如何进入 policy 推理链路。

## 1. 总体结构

`so101_ros2` 不是由一个节点直接同时控制所有设备，而是把系统拆成几层：

```text
真实相机
  -> ROS2 camera node
  -> sensor_msgs/Image topic
  -> policy_runner 订阅

真实 SO-101 follower 机械臂
  -> LeRobot SO101Follower Python API
  -> so101_ros2_bridge Python 节点
  -> ROS2 JointState / joint_commands topic
  -> ros2_control controller
  -> follower 机械臂执行动作
```

其中主要包的职责是：

```text
so101_bringup
  负责 launch，把机械臂、相机、policy、RViz 等节点组合启动。

so101_ros2_bridge
  负责连接 LeRobot 的 SO101Follower / SO101Leader，并把硬件状态转换成 ROS2 topic。

so101_controller
  负责 ros2_control 的 controller_manager、arm_controller、gripper_controller。

so101_hardware_interface
  负责 ros2_control 和 Python 硬件 bridge 之间的 topic 桥接。

so101_teleop
  负责把 leader 或 policy 输出的 JointState 转成 follower controller 命令。
```

## 2. 相机通信方式

相机不是通过 LeRobot robot_client 直接接入，而是通过 ROS2 camera driver 节点接入。

入口文件：

```text
so101_ros2/so101_bringup/launch/include/camera.launch.py
```

它读取：

```text
so101_ros2/so101_bringup/config/so101_cameras.yaml
```

当前配置类似：

```yaml
cameras:
  - name: cam_front
    camera_type: usb_camera
    param_path: so101_usb_cam.yaml
    namespace: follower

  - name: cam_side
    camera_type: realsense2_camera
    param_path: so101_realsense2.yaml
    namespace: static_camera
```

`camera.launch.py` 根据 `camera_type` 选择不同 ROS2 driver：

```text
usb_camera
  -> package: usb_cam
  -> executable: usb_cam_node_exe

realsense2_camera
  -> package: realsense2_camera
  -> executable: realsense2_camera_node
```

所以相机通信链路是：

```text
USB camera / RealSense camera
  -> ROS2 camera driver
  -> /follower/cam_front/image_raw
  -> /static_camera/cam_side/color/image_raw
```

这些图像 topic 再由 policy runner 订阅。


## 2.1 ROS2 camera node 的接口

ROS2 camera node 的接口可以分成三层：

```text
设备层接口：
  /dev/video*
  RealSense USB device

launch/config 接口：
  so101_cameras.yaml
  so101_usb_cam.yaml
  so101_realsense2.yaml

ROS2 topic 接口：
  sensor_msgs/msg/Image
  sensor_msgs/msg/CameraInfo
```

### 2.1.1 launch 层如何描述一台相机

`so101_cameras.yaml` 里的每个条目代表一台相机：

```yaml
- name: cam_front
  camera_type: usb_camera
  param_path: so101_usb_cam.yaml
  namespace: follower
```

字段含义：

```text
name
  ROS2 node 名称，也会参与 topic 名称生成。

camera_type
  选择使用哪种 ROS2 driver。
  当前支持 usb_camera 和 realsense2_camera。

param_path
  对应 driver 的参数文件。

namespace
  ROS2 namespace。
```

所以：

```yaml
name: cam_front
namespace: follower
```

通常会得到类似：

```text
/follower/cam_front/image_raw
/follower/cam_front/camera_info
```

如果是 RealSense：

```yaml
name: cam_side
namespace: static_camera
camera_type: realsense2_camera
```

常见 RGB topic 是：

```text
/static_camera/cam_side/color/image_raw
```

### 2.1.2 USB camera 参数接口

USB 相机的具体设备和图像参数在：

```text
so101_ros2/so101_bringup/config/so101_usb_cam.yaml
```

常用字段：

```yaml
/follower/cam_front:
  ros__parameters:
    video_device: "/dev/video0"
    image_width: 640
    image_height: 480
    framerate: 30.0
    pixel_format: "mjpeg2rgb"
```

这些字段对应关系是：

```text
video_device
  Linux 摄像头设备，例如 /dev/video0、/dev/video2。

image_width / image_height
  发布出来的 ROS Image 分辨率。

framerate
  相机采集频率。

pixel_format
  相机驱动内部解码格式。
```

如果你从 LeRobot 命令迁移过来：

```bash
"cam_main":  {"index_or_path": 2, "width": 424, "height": 240, "fps": 30}
"cam_wrist": {"index_or_path": 0, "width": 424, "height": 240, "fps": 30}
```

那 ROS2 侧可以理解成：

```yaml
/follower/cam_main:
  ros__parameters:
    video_device: "/dev/video2"
    image_width: 424
    image_height: 240
    framerate: 30.0

/follower/cam_wrist:
  ros__parameters:
    video_device: "/dev/video0"
    image_width: 424
    image_height: 240
    framerate: 30.0
```

LeRobot 的 `index_or_path: 2` 和 ROS2 的 `/dev/video2` 在语义上基本对应。

### 2.1.3 相机输出给 policy 的 ROS2 消息

相机 node 发布的是：

```text
sensor_msgs/msg/Image
```

其中 policy adapter 主要使用这些字段：

```text
msg.height
msg.width
msg.encoding
msg.data
```

`so101_ros2_bridge/utils/conversion.py` 里的 `ros_image_to_hwc_uint8()` 支持这些 encoding：

```text
rgb8
bgr8
mono8
```

转换结果是：

```text
np.ndarray
shape: [H, W, 3]
dtype: uint8
color: RGB
```

所以相机接口最终从 ROS2 消息变成模型输入时，是这个过程：

```text
sensor_msgs/Image
  -> ros_image_to_hwc_uint8()
  -> HWC uint8 RGB numpy image
  -> policy observation image
```

这里要特别注意：如果相机 node 发布的 encoding 不是 `rgb8`、`bgr8` 或 `mono8`，当前转换函数会报错。

## 3. Policy 如何拿到相机图像

policy 的输入映射在：

```text
so101_ros2/so101_ros2_bridge/config/policies/io.yaml
```

当前默认类似：

```yaml
observations:
  observation.images.camera1:
    topic: "/follower/cam_front/image_raw"
    msg_type: "sensor_msgs/msg/Image"

  observation.images.camera2:
    topic: "/static_camera/cam_side/color/image_raw"
    msg_type: "sensor_msgs/msg/Image"

  observation.state:
    topic: "/follower/joint_states"
    msg_type: "sensor_msgs/msg/JointState"
```

`policy_runner` 会读取这个 YAML，动态创建 subscriber。

核心文件：

```text
so101_ros2/so101_ros2_bridge/so101_ros2_bridge/policy/runner.py
```

它使用：

```python
ApproximateTimeSynchronizer
```

把多路图像和关节状态按时间戳近似同步。

也就是说，policy 看到的 observation 不是单独一张图，而是一组同步后的消息：

```text
{
  "observation.images.camera1": Image,
  "observation.images.camera2": Image,
  "observation.state": JointState,
}
```

之后具体 policy adapter 再把 ROS2 消息转换成模型输入。例如 `openpi_remote.py` 会把 ROS Image 转成 HWC uint8 numpy 数组，把 JointState 转成 state vector。

## 4. 机械臂通信方式

SO-101 机械臂通信分两部分：

```text
读取机械臂状态
发送目标动作
```

实际和硬件串口通信的是 LeRobot 的 Python API。

核心 bridge 文件：

```text
so101_ros2/so101_ros2_bridge/so101_ros2_bridge/bridge/bridge.py
so101_ros2/so101_ros2_bridge/so101_ros2_bridge/bridge/build.py
```

`build.py` 中创建 follower：

```python
SO101FollowerConfig(
    port=params["port"],
    calibration_dir=Path(params["calibration_dir"]),
    id=params["id"],
    use_degrees=params["use_degrees"],
    max_relative_target=params["max_relative_target"],
    disable_torque_on_disconnect=params["disable_torque_on_disconnect"],
)
```

然后：

```python
SO101Follower(config)
```

所以真正连接机械臂的是：

```text
LeRobot SO101Follower
  -> 串口 /dev/ttyACM*
  -> Feetech motors
```

ROS2 只是通过 bridge 节点把这个 Python API 包装成 topic。


## 4.1 LeRobot SO101Follower Python API 的接口

`SO101Follower` 是真实机械臂的 Python 硬件抽象。它把底层 Feetech 舵机总线包装成几个稳定接口：

```text
SO101FollowerConfig
SO101Follower(config)
robot.connect()
robot.get_observation()
robot.send_action(action)
robot.disconnect()
```

### 4.1.1 构造配置接口

`so101_ros2_bridge/bridge/build.py` 里会用 ROS2 参数构造：

```python
config = SO101FollowerConfig(
    port=params["port"],
    calibration_dir=Path(params["calibration_dir"]),
    id=params["id"],
    use_degrees=params["use_degrees"],
    max_relative_target=params["max_relative_target"],
    disable_torque_on_disconnect=params["disable_torque_on_disconnect"],
)

robot = SO101Follower(config)
```

主要参数含义：

```text
port
  机械臂控制板串口，例如 /dev/ttyACM0。

id
  机器人 ID，也用于选择 calibration 文件。
  例如 id=so101_follower_main 时，通常会找 so101_follower_main.json。

calibration_dir
  calibration JSON 所在目录。

use_degrees
  LeRobot API 读写机械臂时是否使用角度单位。

max_relative_target
  安全限制。目标位置相对当前位置变化太大时会被裁剪。

disable_torque_on_disconnect
  断开连接时是否释放扭矩。
```

注意：当前 `so101_ros2_bridge` 创建 `SO101FollowerConfig` 时没有把 `cameras` 传给 LeRobot。也就是说，在 ROS2 方案里，相机不归 `SO101Follower` 管，而是由 ROS2 camera node 单独管理。

### 4.1.2 连接接口

bridge 初始化时会调用：

```python
self.robot.connect(calibrate=False)
```

这一步会打开串口，并初始化 Feetech 电机总线。

在 LeRobot 原生 `SO101Follower.connect()` 里，如果传 `calibrate=True`，它还可能检查和执行校准；但 `so101_ros2_bridge` 这里传的是 `False`，因此默认依赖已有 calibration。

### 4.1.3 状态读取接口

follower bridge 周期性调用：

```python
obs = self.robot.get_observation()
```

对 `SO101Follower` 来说，这个函数底层会读：

```text
Present_Position
```

返回值是一个 Python dict，典型结构：

```python
{
    "shoulder_pan.pos": 12.3,
    "shoulder_lift.pos": -20.1,
    "elbow_flex.pos": 35.0,
    "wrist_flex.pos": 10.0,
    "wrist_roll.pos": 3.0,
    "gripper.pos": 45.0,
}
```

如果是 LeRobot robot_client 原生模式，并且在 `SO101FollowerConfig` 里配置了 cameras，`get_observation()` 还会附带：

```python
{
    "cam_main": np.ndarray,
    "cam_wrist": np.ndarray,
}
```

但在 `so101_ros2` 当前 bridge 里，`SO101FollowerConfig` 没传 cameras，所以 `get_observation()` 主要负责机械臂关节状态，相机来自 ROS2 camera topic。

### 4.1.4 状态如何变成 ROS2 JointState

LeRobot 返回的 dict 会在 `bridge.py` 里转换成：

```text
sensor_msgs/msg/JointState
```

ROS2 JointState 结构大概是：

```python
JointState(
    name=[
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ],
    position=[...],
    velocity=[...],
)
```

单位转换规则：

```text
普通关节：
  如果 use_degrees=true:
    LeRobot degrees -> ROS radians
  如果 use_degrees=false:
    LeRobot [-100, 100] -> ROS radians

gripper:
  LeRobot [0, 100] -> ROS radians-like value
```

所以 ROS2 其他节点看到的是统一的 `JointState.position`，通常按弧度语义使用。

### 4.1.5 动作下发接口

当 ROS2 control 产生底层命令后，follower bridge 订阅：

```text
/follower/joint_commands
```

消息类型是：

```text
std_msgs/msg/Float64MultiArray
```

callback 中会把 ROS2 的位置命令转换回 LeRobot 需要的格式：

```python
target_positions = {
    "shoulder_pan.pos": ...,
    "shoulder_lift.pos": ...,
    "elbow_flex.pos": ...,
    "wrist_flex.pos": ...,
    "wrist_roll.pos": ...,
    "gripper.pos": ...,
}

self.robot.send_action(target_positions)
```

LeRobot `send_action()` 底层会写：

```text
Goal_Position
```

也就是最终发到 Feetech 舵机的目标位置。

完整接口方向：

```text
读取状态：
  SO101Follower.get_observation()
    -> dict[str, float]
    -> ROS2 JointState

发送动作：
  ROS2 Float64MultiArray
    -> dict[str, float]
    -> SO101Follower.send_action()
    -> Feetech Goal_Position
```

## 5. 机械臂状态发布链路

follower bridge 启动后，会周期性读取机械臂状态：

```python
obs = self.robot.get_observation()
```

然后转换成 ROS2 `sensor_msgs/JointState`。

内部关节顺序是：

```text
shoulder_pan
shoulder_lift
elbow_flex
wrist_flex
wrist_roll
gripper
```

状态发布链路：

```text
真实 SO-101 follower
  -> LeRobot SO101Follower.get_observation()
  -> so101_follower_ros2_bridge
  -> joint_states_raw
  -> ros2_control hardware interface
  -> /follower/joint_states
```

其中：

```text
joint_states_raw
```

是 Python bridge 和 C++ ros2_control hardware interface 之间的内部状态 topic。

`/follower/joint_states` 是 ros2_control / joint_state_broadcaster 对外发布的标准状态 topic。

policy runner 通常订阅的是：

```text
/follower/joint_states
```

## 6. 机械臂动作下发链路

policy 或 leader 输出的动作不会直接调用 `robot.send_action()`，而是先走 ROS2 control 链路。

在 policy 模式下：

```text
policy_runner
  -> 发布 /leader/joint_states
```

这里的 `/leader/joint_states` 可以理解成“虚拟 leader”。它不是来自真实 leader 机械臂，而是来自 policy 输出。

然后：

```text
/leader/joint_states
  -> so101_teleop LeaderTeleopComponent
  -> /follower/arm_controller/joint_trajectory
  -> /follower/gripper_controller/gripper_cmd
```

`LeaderTeleopComponent` 会做两件事：

```text
arm joints
  -> JointTrajectory
  -> arm_controller

gripper joint
  -> GripperCommand action
  -> gripper_controller
```

ros2_control controller 收到目标后，会写入 hardware interface 的 command buffer。

之后 C++ hardware interface 发布：

```text
/follower/joint_commands
```

Python follower bridge 订阅这个 topic，最后调用：

```python
self.robot.send_action(target_positions)
```

真正让机械臂运动。

完整动作链路：

```text
policy output action
  -> /leader/joint_states
  -> LeaderTeleopComponent
  -> /follower/arm_controller/joint_trajectory
  -> /follower/gripper_controller/gripper_cmd
  -> ros2_control controllers
  -> so101_hardware_interface
  -> /follower/joint_commands
  -> so101_follower_ros2_bridge
  -> LeRobot SO101Follower.send_action()
  -> real SO-101 motors
```

## 7. real 模式启动时发生什么

命令：

```bash
ros2 launch so101_bringup so101_teleoperate.launch.py \
  teleop_mode:=real \
  expert:=policy \
  display:=true
```

大致启动顺序：

```text
so101_teleoperate.launch.py
  -> expert=policy
      -> include/policy.launch.py
      -> policy_runner

  -> teleop_mode=real
      -> include/follower.launch.py
      -> follower bridge
      -> controller_manager
      -> arm_controller / gripper_controller

  -> teleop_mode=real
      -> include/camera.launch.py
      -> camera nodes

  -> display=true
      -> RViz

  -> delayed teleop component
      -> LeaderTeleopComponent
```

注意：`policy_runner` 是 lifecycle node。launch 只启动节点，不一定自动开始推理。通常还需要：

```bash
ros2 lifecycle set /policy_runner configure
ros2 lifecycle set /policy_runner activate
```

## 8. 与 LeRobot robot_client 的区别

你之前的 LeRobot 命令类似：

```bash
python -m lerobot.async_inference.robot_client \
  --robot.type so101_follower \
  --robot.port /dev/ttyACM0 \
  --robot.cameras '{"cam_main": ..., "cam_wrist": ...}'
```

这条命令里，LeRobot robot_client 自己做三件事：

```text
1. 打开 /dev/ttyACM0 连接机械臂
2. 打开 OpenCV camera index 2 / 0
3. 连接 LeRobot async inference server
```

而 `so101_ros2` 是：

```text
1. follower bridge 打开 /dev/ttyACM*
2. ROS2 camera node 打开 /dev/video*
3. policy_runner 从 ROS2 topic 收 observation
4. policy adapter 连接推理 server
```

所以不要同时运行：

```text
LeRobot robot_client
so101_ros2 follower bridge / camera nodes
```

否则它们可能会同时抢同一个串口和同一个摄像头设备。

## 9. Topic 总图

policy 模式下最重要的 topic 可以这样理解：

```text
相机输入：
  /follower/cam_front/image_raw
  /static_camera/cam_side/color/image_raw

机械臂状态：
  /follower/joint_states

policy 输出：
  /leader/joint_states

arm controller 输入：
  /follower/arm_controller/joint_trajectory

gripper controller 输入：
  /follower/gripper_controller/gripper_cmd

Python bridge 与 ros2_control 内部通信：
  /follower/joint_states_raw
  /follower/joint_commands
```

## 10. 最重要的配置检查项

运行前建议确认：

```text
1. so101_follower_params.yaml 里的 port 是否是实际 follower 端口。
2. calibration id 是否对应已有校准文件。
3. so101_cameras.yaml 里的相机名称是否和训练时一致。
4. so101_usb_cam.yaml / so101_realsense2.yaml 是否打开了正确设备。
5. io.yaml 的 observation key 是否和 policy adapter 期望一致。
6. observation.state 的 joint 顺序是否和训练数据一致。
7. action 输出维度是否是 6，并对应 SO-101 的 6 个关节。
```

一句话总结：

```text
so101_ros2 通过 ROS2 camera driver 连接相机；
通过 LeRobot SO101Follower/SO101Leader 连接真实机械臂；
通过 ros2_control 把 JointTrajectory / GripperCommand 转成底层 joint_commands；
通过 policy_runner 把相机图像和机械臂状态同步后送给策略，再把策略动作发布回控制链路。
```
