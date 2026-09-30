# so101_ros2 机械臂与图像通信代码整理

这份文件把 `so101_ros2` 中和 **机械臂通信**、**相机/图像通信**、**policy observation 接入** 相关的核心代码摘出来，按调用链整理。

它不是一个可执行单文件，而是源码阅读索引。

## 1. 总链路

### 1.1 机械臂状态链路

```text
真实 SO-101 follower
  -> LeRobot SO101Follower.get_observation()
  -> so101_follower_ros2_bridge
  -> /follower/joint_states_raw
  -> so101_hardware_interface
  -> ros2_control joint_state_broadcaster
  -> /follower/joint_states
  -> policy_runner 订阅
```

### 1.2 机械臂动作链路

```text
policy_runner
  -> /leader/joint_states
  -> so101_teleop LeaderTeleopComponent
  -> /follower/arm_controller/joint_trajectory
  -> /follower/gripper_controller/gripper_cmd
  -> ros2_control controllers
  -> so101_hardware_interface
  -> /follower/joint_commands
  -> so101_follower_ros2_bridge
  -> LeRobot SO101Follower.send_action()
  -> 真实 SO-101 follower
```

### 1.3 图像链路

```text
/dev/video2
  -> usb_cam node
  -> /follower/cam_main/image_raw
  -> policy_runner

/dev/video0
  -> usb_cam node
  -> /follower/cam_wrist/image_raw
  -> policy_runner
```

## 2. launch 启动入口

源文件：

```text
so101_ros2/so101_bringup/launch/so101_teleoperate.launch.py
```

关键代码：

```python
# expert=human 时启动真实 leader
leader_robot_launch = IncludeLaunchDescription(
    PythonLaunchDescriptionSource(
        os.path.join(bringup_pkg, 'launch', 'include', 'leader.launch.py')
    ),
    launch_arguments={
        'model': model,
        'use_sim_time': 'false',
    }.items(),
    condition=IfCondition(EqualsSubstitution(expert, 'human')),
)

# expert=policy 时启动 policy_runner
policy_launch = IncludeLaunchDescription(
    PythonLaunchDescriptionSource(
        os.path.join(bringup_pkg, 'launch', 'include', 'policy.launch.py')
    ),
    condition=IfCondition(EqualsSubstitution(expert, 'policy')),
)

# real 模式启动 follower 机械臂
follower_robot_launch = IncludeLaunchDescription(
    PythonLaunchDescriptionSource(
        os.path.join(bringup_pkg, 'launch', 'include', 'follower.launch.py')
    ),
    launch_arguments={
        'model': model,
        'use_sim_time': 'false',
    }.items(),
    condition=IfCondition(EqualsSubstitution(teleop_mode, 'real')),
)

# real 模式启动相机
cameras_launch = IncludeLaunchDescription(
    PythonLaunchDescriptionSource(
        os.path.join(bringup_pkg, 'launch', 'include', 'camera.launch.py')
    ),
    condition=IfCondition(EqualsSubstitution(teleop_mode, 'real')),
)

# 启动 teleop component，把 /leader/joint_states 转成 follower controller 命令
teleop_launch = IncludeLaunchDescription(
    PythonLaunchDescriptionSource(
        os.path.join(
            get_package_share_directory('so101_teleop'),
            'launch',
            'so101_leader_teleop.launch.py',
        )
    ),
    launch_arguments={
        'mode': teleop_mode,
    }.items(),
)
```

## 3. 机械臂通信代码

### 3.1 follower launch

源文件：

```text
so101_ros2/so101_bringup/launch/include/follower.launch.py
```

关键代码：

```python
robot_launch = IncludeLaunchDescription(
    PythonLaunchDescriptionSource(
        os.path.join(robot_ros2_bridge_pkg, 'launch', 'so101_ros2_bridge.launch.py')
    ),
    launch_arguments={'type': 'follower'}.items(),
)

controller_manager_launch = IncludeLaunchDescription(
    PythonLaunchDescriptionSource(
        os.path.join(controller_pkg, 'launch', 'controller_manager.launch.py')
    ),
    launch_arguments={'type': 'follower'}.items(),
)

spawn_controllers_launch = IncludeLaunchDescription(
    PythonLaunchDescriptionSource(
        os.path.join(controller_pkg, 'launch', 'so101_controllers.launch.py')
    ),
    launch_arguments={'type': 'follower'}.items(),
)
```

这里启动三类东西：

```text
1. Python follower bridge
2. ros2_control controller_manager
3. arm_controller / gripper_controller / joint_state_broadcaster
```

### 3.2 bridge launch

源文件：

```text
so101_ros2/so101_ros2_bridge/launch/so101_ros2_bridge.launch.py
```

关键代码：

```python
robot_type = LaunchConfiguration('type')

config_path = PathJoinSubstitution(
    [
        FindPackageShare('so101_ros2_bridge'),
        'config',
        [
            TextSubstitution(text='so101_'),
            robot_type,
            TextSubstitution(text='_params.yaml'),
        ],
    ]
)

so101_bridge_node = Node(
    package='so101_ros2_bridge',
    executable=[
        robot_type,
        TextSubstitution(text='_ros2_node'),
    ],
    name=[
        TextSubstitution(text='so101_'),
        robot_type,
        TextSubstitution(text='_interface'),
    ],
    output='screen',
    parameters=[
        config_path,
        {'type': robot_type},
    ],
    namespace=robot_type,
)
```

当 `type=follower` 时，实际启动：

```text
package: so101_ros2_bridge
executable: follower_ros2_node
namespace: /follower
params: so101_follower_params.yaml
```

### 3.3 follower 参数

源文件：

```text
so101_ros2/so101_ros2_bridge/config/so101_follower_params.yaml
```

当前配置：

```yaml
so101_follower_ros2_bridge:
  ros__parameters:
    port: "/dev/ttyACM0"
    id: "so101_follower_main"
    # calibration_dir: <abs path to your calibration dir>
    use_degrees: true
    max_relative_target: 10
    disable_torque_on_disconnect: true
    publish_rate: 50.0
```

这些参数最后会进入 LeRobot `SO101FollowerConfig`。

### 3.4 创建 LeRobot SO101Follower

源文件：

```text
so101_ros2/so101_ros2_bridge/so101_ros2_bridge/bridge/build.py
```

关键代码：

```python
from pathlib import Path

from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig

from so101_ros2_bridge.bridge.registry import register_robot


@register_robot('follower')
def create_follower(params: dict):
    config = SO101FollowerConfig(
        port=params['port'],
        calibration_dir=Path(params['calibration_dir']),
        id=params['id'],
        use_degrees=params['use_degrees'],
        max_relative_target=params['max_relative_target'],
        disable_torque_on_disconnect=params['disable_torque_on_disconnect'],
    )
    return SO101Follower(config)
```

这里是 ROS2 参数到 LeRobot 硬件 API 的入口。

### 3.5 bridge 初始化：连接机械臂、创建 topic

源文件：

```text
so101_ros2/so101_ros2_bridge/so101_ros2_bridge/bridge/bridge.py
```

关键代码：

```python
class SO101ROS2Bridge(Node, ABC):
    JOINT_NAMES = [
        'shoulder_pan',
        'shoulder_lift',
        'elbow_flex',
        'wrist_flex',
        'wrist_roll',
        'gripper',
    ]

    def __init__(self, node_name='so101_ros2_bridge'):
        super().__init__(node_name)
        params = self.read_parameters()
        self.use_degrees = params['use_degrees']

        self._joint_state_msg = JointState()
        self._joint_state_msg.name = self.JOINT_NAMES
        self._positions = [0.0] * len(self.JOINT_NAMES)
        self._velocities = [0.0] * len(self.JOINT_NAMES)

        rate = params.get('publish_rate', 30.0)

        self.joint_pub = self.create_publisher(JointState, 'joint_states_raw', 10)
        self.timer = self.create_timer(1.0 / rate, self.publish_joint_states)

        self.declare_parameter('type', 'follower')
        robot_type = self.get_parameter('type').get_parameter_value().string_value
        factory_fn = ROBOT_FACTORY_REGISTRY.get(robot_type)

        self.robot = factory_fn(params)
        self.robot.connect(calibrate=False)
```

这里做了几件关键事：

```text
1. 读取 follower 参数。
2. 创建 /follower/joint_states_raw publisher。
3. 创建定时器，周期性 publish_joint_states。
4. 根据 type=follower 创建 SO101Follower。
5. 调用 robot.connect(calibrate=False) 连接真实机械臂。
```

因为 node 被放在 `/follower` namespace 下，代码里的相对 topic：

```text
joint_states_raw
```

实际就是：

```text
/follower/joint_states_raw
```

### 3.6 读取机械臂状态并发布 JointState

源文件：

```text
so101_ros2/so101_ros2_bridge/so101_ros2_bridge/bridge/bridge.py
```

关键代码：

```python
def publish_joint_states(self):
    try:
        current_time = self.get_clock().now()
        obs = self.get_joints_states()

        for i, joint in enumerate(self.JOINT_NAMES):
            if joint == 'gripper':
                pos = ((obs.get(f'{joint}.pos', 0.0)) / 100.0) * math.pi
            else:
                if self.use_degrees:
                    pos = math.radians(obs.get(f'{joint}.pos', 0.0))
                else:
                    pos = (obs.get(f'{joint}.pos', 0.0) / 100.0) * math.pi
            self._positions[i] = pos

        if self.last_positions is not None:
            dt = (current_time - self.last_time).nanoseconds / 1e9
            if dt > 1e-6:
                for i in range(len(self.JOINT_NAMES)):
                    self._velocities[i] = (self._positions[i] - self.last_positions[i]) / dt

        self._joint_state_msg.header.stamp = self.get_clock().now().to_msg()
        self._joint_state_msg.position = self._positions
        self._joint_state_msg.velocity = self._velocities
        self.joint_pub.publish(self._joint_state_msg)

        self.last_positions = list(self._positions)
        self.last_time = current_time

    except Exception as e:
        self.get_logger().error(f'Failed to publish joint states: {e}')
```

对应 follower 的状态读取实现：

```python
class FollowerBridge(SO101ROS2Bridge):
    def get_joints_states(self) -> dict:
        return self.robot.get_observation()
```

所以状态读取本质是：

```text
LeRobot SO101Follower.get_observation()
  -> dict: {"shoulder_pan.pos": ..., ...}
  -> ROS2 JointState
  -> /follower/joint_states_raw
```

### 3.7 接收底层 joint_commands 并发送给机械臂

源文件：

```text
so101_ros2/so101_ros2_bridge/so101_ros2_bridge/bridge/bridge.py
```

关键代码：

```python
class FollowerBridge(SO101ROS2Bridge):
    def __init__(self):
        super().__init__('so101_follower_ros2_bridge')

        self.create_subscription(
            Float64MultiArray,
            'joint_commands',
            self.command_callback,
            qos_joint_cmds,
        )

    def command_callback(self, msg: Float64MultiArray):
        if len(msg.data) != len(self.JOINT_NAMES):
            self.get_logger().error(
                f'Received command with {len(msg.data)} joints, but expected {len(self.JOINT_NAMES)}.'
            )
            return

        target_positions = {}
        for i, joint in enumerate(self.JOINT_NAMES):
            target_positions[f'{joint}.pos'] = self.radians_to_normalized(joint, msg.data[i])

        try:
            self.robot.send_action(target_positions)
        except Exception as e:
            self.get_logger().error(f'Failed to send commands to robot: {e}')
```

因为 node 在 `/follower` namespace 下，所以相对 topic：

```text
joint_commands
```

实际是：

```text
/follower/joint_commands
```

这里完成：

```text
ROS2 Float64MultiArray
  -> {"shoulder_pan.pos": ..., ...}
  -> SO101Follower.send_action()
  -> Feetech Goal_Position
```

### 3.8 ros2_control hardware interface 桥接代码

源文件：

```text
so101_ros2/so101_hardware_interface/src/so101_hardware_bridge.cpp
```

关键代码：

```cpp
hardware_interface::CallbackReturn So101HardwareBridge::on_activate(
  const rclcpp_lifecycle::State &)
{
  cmd_publisher_ =
    node_->create_publisher<std_msgs::msg::Float64MultiArray>("joint_commands", 10);

  state_subscriber_ = node_->create_subscription<sensor_msgs::msg::JointState>(
    "joint_states_raw", 10,
    std::bind(&So101HardwareBridge::state_subscriber_callback, this, std::placeholders::_1));

  return hardware_interface::CallbackReturn::SUCCESS;
}
```

读取状态：

```cpp
hardware_interface::return_type So101HardwareBridge::read(
  const rclcpp::Time &,
  const rclcpp::Duration &)
{
  rclcpp::spin_some(node_);
  return hardware_interface::return_type::OK;
}
```

写入命令：

```cpp
hardware_interface::return_type So101HardwareBridge::write(
  const rclcpp::Time &,
  const rclcpp::Duration &)
{
  auto msg = std::make_unique<std_msgs::msg::Float64MultiArray>();

  for (size_t i = 0; i < hw_commands_.size(); ++i) {
    if (std::isnan(hw_commands_[i])) {
      std::lock_guard<std::mutex> guard(lock_);
      msg->data.push_back(hw_positions_[i]);
    } else {
      msg->data.push_back(hw_commands_[i]);
    }
  }

  cmd_publisher_->publish(std::move(msg));

  return hardware_interface::return_type::OK;
}
```

接收 Python bridge 发来的状态：

```cpp
void So101HardwareBridge::state_subscriber_callback(
  const sensor_msgs::msg::JointState::SharedPtr msg)
{
  std::lock_guard<std::mutex> guard(lock_);

  for (size_t i = 0; i < msg->name.size(); ++i) {
    for (size_t j = 0; j < joint_names_.size(); ++j) {
      if (msg->name[i] == joint_names_[j]) {
        hw_positions_[j] = msg->position[i];
        if (msg->velocity.size() > i) {
          hw_velocities_[j] = msg->velocity[i];
        }
        break;
      }
    }
  }
}
```

这个 C++ 插件是 ros2_control 与 Python follower bridge 的中间层。

### 3.9 controller 配置

源文件：

```text
so101_ros2/so101_controller/config/so101_controllers.yaml
```

关键配置：

```yaml
/follower:
  controller_manager:
    ros__parameters:
      update_rate: 50

      arm_controller:
        type: joint_trajectory_controller/JointTrajectoryController

      gripper_controller:
        type: position_controllers/GripperActionController

      joint_state_broadcaster:
        type: joint_state_broadcaster/JointStateBroadcaster

  arm_controller:
    ros__parameters:
      update_rate: 25
      joints:
        - shoulder_pan
        - shoulder_lift
        - elbow_flex
        - wrist_flex
        - wrist_roll
      command_interfaces:
        - position
      state_interfaces:
        - position
        - velocity

  gripper_controller:
    ros__parameters:
      joint: gripper
```

arm controller 控制前 5 个关节，gripper controller 单独控制夹爪。

### 3.10 policy/leader JointState 到 follower controller

源文件：

```text
so101_ros2/so101_teleop/src/leader_teleop_component.cpp
```

关键代码：

```cpp
auto leader_topic = this->declare_parameter<std::string>(
  "leader_joint_states_topic",
  "/leader/joint_states");

auto follower_arm_topic = this->declare_parameter<std::string>(
  "follower_trajectory_topic",
  "/follower/arm_controller/joint_trajectory");

auto follower_gripper_action = this->declare_parameter<std::string>(
  "follower_gripper_action_name",
  "/follower/gripper_controller/gripper_cmd");

trajectory_pub_ =
  this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
    follower_arm_topic,
    traj_pub_qos);

gripper_action_client_ =
  rclcpp_action::create_client<control_msgs::action::GripperCommand>(
    this,
    follower_gripper_action);

joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
  leader_topic, js_sub_qos,
  std::bind(&LeaderTeleopComponent::joint_state_callback, this, std::placeholders::_1));
```

callback 中把 arm joints 转成 trajectory：

```cpp
trajectory_msg_.points.clear();
trajectory_msg_.header.stamp = this->get_clock()->now();
trajectory_msg_.joint_names = ordered_arm_joint_names_;

trajectory_msgs::msg::JointTrajectoryPoint point;
point.positions.reserve(ordered_arm_joint_names_.size());
for (size_t i = 0; i < msg->name.size(); ++i) {
  if (leader_gripper_joint_index_.has_value() && i == *leader_gripper_joint_index_) {
    continue;
  }
  point.positions.push_back(msg->position[i]);
}

point.time_from_start = rclcpp::Duration(std::chrono::milliseconds(3));
trajectory_msg_.points.push_back(point);
trajectory_pub_->publish(trajectory_msg_);
```

夹爪转成 gripper action：

```cpp
if (leader_gripper_joint_index_.has_value()) {
  double current_gripper_pos = msg->position[*leader_gripper_joint_index_];

  if (std::abs(current_gripper_pos - last_gripper_goal_position_) > gripper_deadband_) {
    auto goal_msg = control_msgs::action::GripperCommand::Goal();
    goal_msg.command.position = current_gripper_pos;
    goal_msg.command.max_effort = 10.0;

    auto send_goal_options =
      rclcpp_action::Client<control_msgs::action::GripperCommand>::SendGoalOptions();
    gripper_action_client_->async_send_goal(goal_msg, send_goal_options);

    last_gripper_goal_position_ = current_gripper_pos;
  }
}
```

## 4. 图像通信代码

### 4.1 相机列表配置

源文件：

```text
so101_ros2/so101_bringup/config/so101_cameras.yaml
```

当前配置：

```yaml
cameras:
  - name: cam_main
    camera_type: usb_camera
    param_path: so101_usb_cam.yaml
    namespace: follower

  - name: cam_wrist
    camera_type: usb_camera
    param_path: so101_usb_cam.yaml
    namespace: follower
```

这个文件说明有两台相机：

```text
cam_main
cam_wrist
```

都用 `usb_cam` driver，并且都在 `/follower` namespace 下。

### 4.2 USB 相机参数

源文件：

```text
so101_ros2/so101_bringup/config/so101_usb_cam.yaml
```

当前配置：

```yaml
/**:
  ros__parameters:
    framerate: 30.0
    io_method: "mmap"
    pixel_format: "mjpeg2rgb"
    image_width: 424
    image_height: 240

/follower/cam_main:
  ros__parameters:
    video_device: "/dev/video2"
    frame_id: "cam_main"
    camera_name: "cam_main"

/follower/cam_wrist:
  ros__parameters:
    video_device: "/dev/video0"
    frame_id: "cam_wrist"
    camera_name: "cam_wrist"
```

对应关系：

```text
/dev/video2 -> /follower/cam_main/image_raw
/dev/video0 -> /follower/cam_wrist/image_raw
```

### 4.3 camera.launch.py 动态创建 camera node

源文件：

```text
so101_ros2/so101_bringup/launch/include/camera.launch.py
```

driver registry：

```python
TYPE_REGISTRY: Dict[str, Tuple[str, str]] = {
    'usb_camera': ('usb_cam', 'usb_cam_node_exe'),
    'realsense2_camera': ('realsense2_camera', 'realsense2_camera_node'),
}
```

配置结构：

```python
@dataclass
class CameraSpec:
    name: str
    camera_type: str
    param_path: Path
    namespace: Optional[str] = None
    remappings: Optional[List[Tuple[str, str]]] = None
```

读取 YAML：

```python
def parse_cameras_config(
    config_file_rel: str = 'config/so101_cameras.yaml',
) -> List[CameraSpec]:
    cfg = load_yaml_from_pkg(PKG_BRINGUP, config_file_rel)
    if not cfg or 'cameras' not in cfg:
        raise RuntimeError(f"No valid 'cameras' list found in {config_file_rel}")

    cams: List[CameraSpec] = []
    for cam in cfg['cameras']:
        name = cam.get('name')
        cam_type = cam.get('camera_type')
        rel_param = cam.get('param_path')
        namespace = cam.get('namespace')
        remappings = cam.get('remappings')

        if cam_type not in TYPE_REGISTRY:
            raise ValueError(
                f"Unknown camera type '{cam_type}'. Supported: {list(TYPE_REGISTRY.keys())}"
            )

        cams.append(
            CameraSpec(
                name=name,
                camera_type=cam_type,
                param_path=_resolve_param_path(rel_param),
                namespace=namespace,
                remappings=remappings,
            )
        )
    return cams
```

构造 ROS2 camera node：

```python
def build_node_for_camera(spec: CameraSpec) -> Node:
    pkg, exe = TYPE_REGISTRY[spec.camera_type]

    overrides: Dict[str, Any] = {
        'camera_name': spec.name,
    }

    if spec.camera_type == 'usb_camera':
        overrides.setdefault('frame_id', spec.name)

    parameters = [str(spec.param_path), overrides]

    default_usb_cam_remaps = [
        ('image_raw', f'{spec.name}/image_raw'),
        ('image_raw/compressed', f'{spec.name}/image_compressed'),
        ('image_raw/compressedDepth', f'{spec.name}/compressedDepth'),
        ('image_raw/theora', f'{spec.name}/image_raw/theora'),
        ('camera_info', f'{spec.name}/camera_info'),
    ]

    remappings = spec.remappings
    if remappings is None and spec.camera_type == 'usb_camera':
        remappings = default_usb_cam_remaps

    return Node(
        package=pkg,
        executable=exe,
        name=spec.name,
        namespace=spec.namespace,
        output='screen',
        parameters=parameters,
        remappings=remappings or [],
    )
```

启动所有 camera node：

```python
def generate_launch_description():
    camera_specs = parse_cameras_config()
    nodes = [build_node_for_camera(spec) for spec in camera_specs]
    return LaunchDescription([GroupAction(nodes)])
```

## 5. policy_runner 如何订阅图像和关节状态

### 5.1 policy I/O 映射

源文件：

```text
so101_ros2/so101_ros2_bridge/config/policies/io.yaml
```

当前配置：

```yaml
observations:
  observation.images.cam_main:
    topic: "/follower/cam_main/image_raw"
    msg_type: "sensor_msgs/msg/Image"

  observation.images.cam_wrist:
    topic: "/follower/cam_wrist/image_raw"
    msg_type: "sensor_msgs/msg/Image"

  observation.state:
    topic: "/follower/joint_states"
    msg_type: "sensor_msgs/msg/JointState"
    names:
      - "shoulder_pan"
      - "shoulder_lift"
      - "elbow_flex"
      - "wrist_flex"
      - "wrist_roll"
      - "gripper"

action:
  topic: "/leader/joint_states"
  msg_type: "sensor_msgs/msg/JointState"
  names:
    - "shoulder_pan"
    - "shoulder_lift"
    - "elbow_flex"
    - "wrist_flex"
    - "wrist_roll"
    - "gripper"
```

### 5.2 创建 subscriber 和同步器

源文件：

```text
so101_ros2/so101_ros2_bridge/so101_ros2_bridge/policy/runner.py
```

读取 I/O YAML：

```python
def _load_obs_and_action_cfg(self) -> tuple[Dict[str, Any], Dict[str, Any]]:
    cfg_path = POLICY_BASE_DIR / 'io.yaml'

    with cfg_path.open('r') as f:
        data = yaml.safe_load(f) or {}

    obs_cfg = data.get('observations')
    action_cfg = data.get('action')
    return obs_cfg, action_cfg
```

创建同步 callback：

```python
def _make_sync_cb(self, keys: List[str]):
    def cb(*msgs):
        if len(msgs) != len(keys):
            self.get_logger().warn(
                f'Sync callback got {len(msgs)} msgs but expected {len(keys)}'
            )
            return
        with self._obs_lock:
            self._latest_msgs = {k: m for k, m in zip(keys, msgs)}

    return cb
```

根据 `io.yaml` 创建 subscriber：

```python
obs_keys = list(self._obs_cfg.keys())
subscribers = []

for key in obs_keys:
    entry = obs_cfg[key]
    topic = entry['topic']
    msg_type_str = entry['msg_type']
    msg_cls = resolve_msg_type(msg_type_str)
    subscribers.append(
        Subscriber(self, msg_cls, topic, qos_profile=qos_profile_sensor_data)
    )
```

同步多路图像和关节状态：

```python
self._sync = ApproximateTimeSynchronizer(
    subscribers,
    queue_size=queue_size,
    slop=slop,
)
self._sync.registerCallback(self._make_sync_cb(obs_keys))
```

同步后的缓存结构：

```python
self._latest_msgs = {
    "observation.images.cam_main": Image,
    "observation.images.cam_wrist": Image,
    "observation.state": JointState,
}
```

### 5.3 policy_runner 发布 action

源文件：

```text
so101_ros2/so101_ros2_bridge/so101_ros2_bridge/policy/runner.py
```

创建 action publisher：

```python
act_topic = action_entry['topic']
act_msg_type_str = action_entry['msg_type']
act_msg_cls = resolve_msg_type(act_msg_type_str)

self._cmd_pub = self.create_publisher(act_msg_cls, act_topic, 10)
```

发布下一步动作：

```python
def _publish_step(self) -> None:
    if self._cmd_pub is None or self._policy is None:
        return

    action = self._policy.get_action()
    if action is None:
        return

    current_pos = list(action)
    n = len(current_pos)

    js_cmd = JointState()
    js_cmd.header.stamp = self.get_clock().now().to_msg()
    js_cmd.name = self._action_joint_names[:n]
    js_cmd.position = current_pos

    self._cmd_pub.publish(js_cmd)
```

当前 `io.yaml` 中 action topic 是：

```text
/leader/joint_states
```

所以 policy 输出会作为“虚拟 leader JointState”进入 teleop 组件。

### 5.4 policy_runner 推理入口

源文件：

```text
so101_ros2/so101_ros2_bridge/so101_ros2_bridge/policy/runner.py
```

关键代码：

```python
def _inference_step(self) -> None:
    if self._policy is None:
        return

    with self._obs_lock:
        ros_obs = self._latest_msgs

    if ros_obs is None:
        return

    delay = self._inference_delay if self._use_delay_compensation else 0.0

    self._policy.infer(
        ros_obs=ros_obs,
        time_per_action=self._publish_period,
        inference_delay=delay,
    )
```

## 6. ROS 图像和 JointState 到模型输入的转换代码

源文件：

```text
so101_ros2/so101_ros2_bridge/so101_ros2_bridge/utils/conversion.py
```

### 6.1 ROS Image 转 numpy

```python
def ros_image_to_hwc_uint8(msg: Image) -> np.ndarray:
    """Convert a ROS image message to HWC uint8 (0–255)."""
    enc = msg.encoding
    if enc not in ('rgb8', 'bgr8', 'mono8'):
        raise ValueError(f'Unsupported encoding: {enc}')
    ch = 3 if enc in ('rgb8', 'bgr8') else 1
    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, ch)
    if enc == 'bgr8':
        arr = arr[..., ::-1]
    if ch == 1:
        arr = np.repeat(arr, 3, axis=2)
    return arr.astype(np.uint8)
```

输入：

```text
sensor_msgs/msg/Image
```

输出：

```text
np.ndarray
shape: [H, W, 3]
dtype: uint8
color: RGB
```

### 6.2 ROS JointState 转 state vector

```python
def ros_jointstate_to_vec(
    js_msg: JointState,
    joint_order: Optional[List[str]] = None,
    use_lerobot_ranges_norms: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    pos = list(getattr(js_msg, 'position', []))
    names = list(getattr(js_msg, 'name', []))

    if joint_order:
        name_to_idx = {n: i for i, n in enumerate(names)}
        try:
            vals = [pos[name_to_idx[n]] for n in joint_order]
        except KeyError as e:
            missing = set(joint_order) - set(names)
            raise KeyError(f'Joint(s) {missing} not found in JointState.name') from e

        if use_lerobot_ranges_norms:
            vals = [radians_to_normalized(j_name, v) for j_name, v in zip(joint_order, vals)]

        out = np.asarray(vals, dtype=np.float32)
        joint_names = list(joint_order)
    else:
        if not pos:
            raise ValueError('JointState.position is empty')

        joint_names = names if names else [f'joint_{i}' for i in range(len(pos))]
        vals = pos

        if use_lerobot_ranges_norms:
            vals = [radians_to_normalized(j_name, v) for j_name, v in zip(joint_names, vals)]

        out = np.asarray(vals, dtype=np.float32)

    return out, joint_names
```

这里依赖 `io.yaml` 或 robot properties 中的 joint order，保证 state 向量顺序稳定。

## 7. OpenPI remote policy 如何使用图像和关节状态

源文件：

```text
so101_ros2/so101_ros2_bridge/so101_ros2_bridge/policy/openpi_remote.py
```

读取配置：

```python
extra = cfg.extra or {}
server_cfg = extra.get("server", {})
obs_cfg = extra.get("observation", {})
lpf_cfg = extra.get("lpf_filtering", {})

host = server_cfg.get("host", "127.0.0.1")
port = int(server_cfg.get("port", 8080))

self._base_image_key = obs_cfg.get("base_image_key", "observation.images.cam_main")
self._wrist_image_key = obs_cfg.get("wrist_image_key", "observation.images.cam_wrist")
self._state_key = obs_cfg.get("state_key", "observation.state")
self._task = cfg.task
```

连接 OpenPI server：

```python
self._client = WebsocketClientPolicy(host=host, port=port)
```

把 ROS observation 转成 OpenPI observation：

```python
def make_observation(self, ros_obs: Mapping[str, Any]) -> dict:
    base_img = ros_image_to_hwc_uint8(ros_obs[self._base_image_key])
    wrist_img = ros_image_to_hwc_uint8(ros_obs[self._wrist_image_key])
    state, _ = ros_jointstate_to_vec(
        ros_obs[self._state_key],
        joint_order=self._state_joint_names,
        use_lerobot_ranges_norms=False,
    )

    return {
        "observation/image": base_img,
        "observation/wrist_image": wrist_img,
        "observation/state": state.astype(np.float32),
        "prompt": self._task,
    }
```

调用远程推理并写入 action buffer：

```python
def predict_action_chunk(
    self,
    observation: dict,
    time_per_action: float,
    inference_delay: float,
) -> None:
    result = self._client.infer(observation)
    actions = np.asarray(result["actions"], dtype=np.float32)

    if actions.ndim != 2:
        self.node.get_logger().error(
            f"[OpenPIRemote] Expected actions shape [T, D], got {actions.shape}"
        )
        return

    actions_list = actions.tolist()
    if not actions_list:
        self.node.get_logger().warn("[OpenPIRemote] Empty action chunk returned.")
        return

    if inference_delay > 0.0 and time_per_action > 0.0:
        start_index = math.ceil(inference_delay / time_per_action)
    else:
        start_index = 0

    self._action_buffer.set(actions_list, start_index=start_index)
```

每次发布动作时：

```python
def get_action(self) -> Optional[List[float]]:
    return self._action_buffer.get()
```

## 8. 当前关键 topic 对照

```text
图像输入：
  /follower/cam_main/image_raw
  /follower/cam_wrist/image_raw

机械臂状态：
  /follower/joint_states_raw
  /follower/joint_states

policy observation keys：
  observation.images.cam_main
  observation.images.cam_wrist
  observation.state

policy action output：
  /leader/joint_states

teleop 到 controller：
  /follower/arm_controller/joint_trajectory
  /follower/gripper_controller/gripper_cmd

ros2_control 到 Python bridge：
  /follower/joint_commands
```

## 9. 最小源码调用关系

```text
相机：
  so101_cameras.yaml
    -> camera.launch.py
    -> usb_cam_node_exe
    -> /follower/cam_main/image_raw
    -> /follower/cam_wrist/image_raw
    -> policy/runner.py Subscriber
    -> conversion.py ros_image_to_hwc_uint8
    -> openpi_remote.py make_observation

机械臂状态：
  so101_follower_params.yaml
    -> so101_ros2_bridge.launch.py
    -> follower_ros2_node
    -> bridge/build.py create_follower
    -> SO101Follower.get_observation
    -> bridge.py publish_joint_states
    -> /follower/joint_states_raw
    -> so101_hardware_bridge.cpp
    -> /follower/joint_states
    -> policy/runner.py Subscriber

机械臂动作：
  openpi_remote.py get_action
    -> policy/runner.py _publish_step
    -> /leader/joint_states
    -> leader_teleop_component.cpp
    -> /follower/arm_controller/joint_trajectory
    -> /follower/gripper_controller/gripper_cmd
    -> ros2_control controllers
    -> so101_hardware_bridge.cpp write
    -> /follower/joint_commands
    -> bridge.py command_callback
    -> SO101Follower.send_action
```
