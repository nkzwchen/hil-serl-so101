# 基于 so101_ros2 学习 ROS2 基础

这份笔记把你当作已经掌握计算机基础、但刚开始接触 ROS2 的读者，用 `so101_ros2` 这个真实项目来解释 ROS2 的核心概念。

你可以先把 ROS2 理解成一个机器人软件通信系统：很多小程序各自负责一件事，它们通过 topic、service、action、参数和 launch 组合起来。

## 1. Node：一个节点就是一个小程序

在 `so101_ros2` 里，不是一个大程序控制所有东西，而是很多节点协作：

```text
相机节点
  负责读取摄像头，发布图像

follower bridge 节点
  负责连接真实 SO-101 机械臂，读状态、发动作

policy_runner 节点
  负责接收图像和关节状态，调用策略模型，输出动作

teleop component
  负责把 /leader/joint_states 转成 follower controller 命令

controller_manager
  负责 ros2_control 控制器管理
```

对应代码例子：

```text
so101_ros2_bridge/follower_ros2_node.py
so101_ros2_bridge/policy_runner_ros2_node.py
so101_bringup/launch/include/camera.launch.py
```

一句话：**ROS2 node 是系统里的功能模块。**

## 2. Topic：节点之间的数据管道

ROS2 最核心的概念之一是 topic。

一个节点可以发布 topic，另一个节点可以订阅 topic。

相机例子：

```text
usb_cam node
  -> 发布 /follower/cam_main/image_raw

policy_runner
  -> 订阅 /follower/cam_main/image_raw
```

机械臂状态例子：

```text
follower bridge
  -> /follower/joint_states_raw
  -> ros2_control
  -> /follower/joint_states
```

policy runner 会订阅：

```text
/follower/joint_states
```

所以 policy 的 observation 主要来自：

```text
/follower/cam_main/image_raw
/follower/cam_wrist/image_raw
/follower/joint_states
```

一句话：**topic 就是 ROS2 里的数据流通道。**

## 3. Message：topic 上传输的数据格式

topic 不是随便传 Python dict，而是传 ROS2 定义好的 message 类型。

在 `so101_ros2` 里常见：

```text
sensor_msgs/msg/Image
  图像消息

sensor_msgs/msg/JointState
  关节状态消息

std_msgs/msg/Float64MultiArray
  一组 float 数字，常用于底层关节命令

trajectory_msgs/msg/JointTrajectory
  机械臂轨迹命令

control_msgs/action/GripperCommand
  夹爪控制 action
```

例如 `/follower/joint_states` 的类型是：

```text
sensor_msgs/msg/JointState
```

它里面主要有：

```text
name
  关节名字

position
  关节位置

velocity
  关节速度
```

比如：

```text
name = ["shoulder_pan", "shoulder_lift", ..., "gripper"]
position = [0.1, -0.2, ..., 0.5]
```

## 4. Publisher 和 Subscriber

在代码里，发布 topic 叫 publisher，订阅 topic 叫 subscriber。

例如 follower bridge 发布机械臂状态：

```python
self.joint_pub = self.create_publisher(JointState, 'joint_states_raw', 10)
```

因为节点在 `/follower` namespace 下，所以实际 topic 是：

```text
/follower/joint_states_raw
```

接收命令：

```python
self.create_subscription(
    Float64MultiArray,
    'joint_commands',
    self.command_callback,
    qos_joint_cmds,
)
```

实际订阅：

```text
/follower/joint_commands
```

所以一发一收就是：

```text
publisher -> topic -> subscriber callback
```

## 5. Namespace：给 topic 和 node 加前缀

`so101_ros2` 里大量使用 namespace。

比如 follower bridge 启动时：

```python
namespace=robot_type
```

当 `robot_type = follower` 时：

```text
joint_states_raw
```

会变成：

```text
/follower/joint_states_raw
```

相机也一样：

```yaml
name: cam_main
namespace: follower
```

最后 topic 类似：

```text
/follower/cam_main/image_raw
```

namespace 的作用是避免多个机器人、多个设备 topic 名字冲突。

## 6. Launch：一次启动多个节点

你运行的：

```bash
ros2 launch so101_bringup so101_teleoperate.launch.py \
  teleop_mode:=real \
  expert:=policy \
  display:=true
```

这不是启动一个程序，而是启动一组节点。

大概是：

```text
so101_teleoperate.launch.py
  -> follower.launch.py
  -> camera.launch.py
  -> policy.launch.py
  -> so101_leader_teleop.launch.py
  -> display.launch.py
```

参数含义：

```text
teleop_mode:=real
  使用真实硬件。

expert:=policy
  不使用真实 leader 机械臂，而是让 policy 输出动作。

display:=true
  启动 RViz 可视化。
```

## 7. Parameter：运行时配置

ROS2 节点通常通过参数配置行为。

比如 follower 机械臂参数在：

```text
so101_ros2_bridge/config/so101_follower_params.yaml
```

里面有：

```yaml
port: "/dev/ttyACM0"
id: "so101_follower_main"
publish_rate: 50.0
```

这告诉 follower bridge：

```text
从 /dev/ttyACM0 连接机械臂
使用 so101_follower_main 对应 calibration
以 50Hz 发布状态
```

policy 参数在：

```text
so101_ros2_bridge/config/so101_policy_params.yaml
```

比如：

```yaml
policy_name: "openpi_remote"
publish_rate: 25.0
inference_rate: 1.0
task: "pick up the orange"
```

## 8. QoS：通信质量设置

QoS 可以理解成 topic 通信策略。

相机数据通常是：

```text
低延迟优先
可以丢旧帧
不用保证每一帧都到
```

所以 policy runner 用：

```python
qos_profile_sensor_data
```

订阅图像。

而机械臂命令更倾向可靠：

```python
QoSReliabilityPolicy.RELIABLE
```

因为命令消息不能太随意丢。

## 9. Timer：周期性执行任务

ROS2 里 timer 用来定频运行。

比如 follower bridge：

```python
self.timer = self.create_timer(1.0 / rate, self.publish_joint_states)
```

含义是：每隔 `1/rate` 秒读取一次机械臂状态并发布。

policy runner 里有两个 timer：

```text
inference_timer
  低频推理

publish_timer
  高频发布动作
```

所以它可以做到：

```text
模型每秒推理 1 次
机械臂每秒执行 25 步动作
```

## 10. message_filters：同步多路传感器

policy 不只需要一张图，还需要：

```text
主相机图像
腕部相机图像
关节状态
```

这些消息到达时间不完全一样，所以 `policy_runner` 使用：

```python
ApproximateTimeSynchronizer
```

把它们按时间戳近似同步。

同步之后得到：

```python
{
    "observation.images.cam_main": Image,
    "observation.images.cam_wrist": Image,
    "observation.state": JointState,
}
```

然后送给 policy。

## 11. ros2_control：标准控制框架

`so101_ros2` 没有直接让 policy 调 `robot.send_action()`，而是走了 ROS2 控制链路：

```text
policy_runner
  -> /leader/joint_states
  -> teleop component
  -> /follower/arm_controller/joint_trajectory
  -> ros2_control
  -> /follower/joint_commands
  -> follower bridge
  -> robot.send_action()
```

`ros2_control` 的好处是：控制器、状态发布、硬件接口都比较标准化。

这里有：

```text
arm_controller
  控制 shoulder_pan 到 wrist_roll 这 5 个 arm joints

gripper_controller
  控制 gripper

joint_state_broadcaster
  发布标准 /follower/joint_states
```

## 12. Lifecycle Node：可配置、可启动、可停止的节点

`policy_runner` 是 lifecycle node。

它不是一启动就开始推理，而是要经过状态切换：

```bash
ros2 lifecycle set /policy_runner configure
ros2 lifecycle set /policy_runner activate
```

`configure` 阶段做：

```text
读取参数
加载 policy
创建 subscriber
创建 publisher
```

`activate` 阶段做：

```text
启动 inference timer
启动 publish timer
```

这样更安全，因为策略控制机械臂前可以先确认系统准备好了。

## 13. 用 so101_ros2 理解 ROS2

`so101_ros2` 里的 ROS2 可以这样理解：

```text
相机 driver 把图像发布成 Image topic；
follower bridge 把真实机械臂状态发布成 JointState topic；
policy_runner 同步图像和关节状态，生成动作；
teleop 和 ros2_control 把动作转换成底层 joint_commands；
follower bridge 收到底层命令后调用 LeRobot API 驱动真实机械臂。
```

最核心的 ROS2 基础就是：

```text
Node
  一个功能模块

Topic
  模块之间的数据管道

Message
  topic 上的数据格式

Launch
  一次启动多个节点

Parameter
  节点配置

Namespace
  topic/node 名前缀

Timer
  周期性执行

QoS
  通信策略

Lifecycle
  节点状态管理

ros2_control
  标准机器人控制框架
```

读 ROS2 项目时，可以反复问这几个问题：

```text
这个节点是谁？
它订阅什么？
它发布什么？
消息类型是什么？
参数从哪里来？
launch 谁启动了它？
```

把这些问题串起来，大多数 ROS2 项目的结构就能读懂。
