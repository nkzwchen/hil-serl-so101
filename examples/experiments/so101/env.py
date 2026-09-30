"""SO101 的 gym 环境（HIL-SERL 用）。

设计要点（已与使用者确认）：
  · 命令路径：直接发 /follower/arm_controller/joint_trajectory + gripper action
              （env 是 follower 的唯一控制源，避免与 teleop 组件抢话题）
  · 观测源  ：/follower/joint_states（ros2_control 广播，io.yaml 同款）
              + /follower/cam_{main,wrist}/image_raw
  · 动作空间：**3 维**关节增量，action ∈ [-1,1]^3，依次对应
              shoulder_lift / elbow_flex / wrist_flex。
              ★ shoulder_pan / wrist_roll / gripper 全部**冻结**，
                **不进动作向量**（定义见 so101_home_pose.FROZEN_JOINTS）——
                策略和人手在结构上就动不了它们。
              ⚠️ 冻结 pan 的代价：末端被限制在一个竖直平面内，
                 只剩 2 个平移自由度（实测点云 s3/s1 ≈ 1.5e-15）。
  · 不用 RelativeFrame：它变换的是笛卡尔量（tcp_pose/tcp_vel），
              关节增量没有坐标系，变换无意义。关节增量本身就是相对量。
  · 力矩管理：/follower/set_torque、/leader/set_torque（L1 已实现）

观测空间结构（必须符合 SERLObsWrapper 的契约）：
    {"state": {...proprio...}, "images": {...}}
  SERLObsWrapper 会把 images 展开到顶层、把 state 拍平。

线程模型：
  · 后台线程跑 rclpy spin，只负责把最新消息写进内存缓存（加锁）
  · 主线程按 hz 读缓存、发命令、算 reward
    这样就得到「同步 step」的语义，而不依赖 ROS2 回调时序

前提：外部已经跑着 follower + leader 的桥（以及 ros2_control）。
      本类只创建自己的节点，不负责启动别人。
"""

from __future__ import annotations

import math
import os
import sys
import threading
import time
from typing import Optional

import gymnasium as gym
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image, JointState
from control_msgs.action import GripperCommand
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import SetBool
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import cv2

# ── 两臂归位姿态 / 主臂抬高偏置的**唯一真相源** ──────────────────────────
# 主臂那张桌子比从臂这边略高，所以**两臂的家姿态不一样**：主臂要抬得高一些，
# 否则夹爪贴着自己的桌面 → 堵转 → 舵机过热保护（真机事故，见 安全边界-主臂.md §0）。
#
# ★★ 由此产生的强制约定（本文件两个地方必须遵守）★★
#       follower_pose  =  leader_pose  −  LEADER_HOME_OFFSET
#    · 模式 A（主臂跟随从臂）：发给主臂的目标 = 从臂位置 **+ 偏置**
#    · 模式 B（从臂跟随主臂）：从臂目标     = 主臂位置 **− 偏置**
#   漏掉任何一处，偏置就会被当成"人工输入"，从臂会莫名其妙地抬高/压低。
#   （同一类 bug 在 phase3_cartesian_teleop.py 里也踩过：软启动权重衰减到 0
#     = 从臂直接用主臂绝对角，等价于把偏置当人工输入。）
# ⚠️ 必须把本文件所在目录加进 sys.path，否则 `import so101_home_pose` 会
#    ModuleNotFoundError：真实训练路径是 `import experiments.so101.env`，
#    此时只有 examples/ 与 experiments/ 在 sys.path 上，**不含 so101/**。
#    （踩坑 2026-09-29：加了这行 import 后 config.py 的 fake_env 直接 import 失败。）
#    用 sys.path 而不是相对 import，是为了让 env.py 被当脚本直接跑时也能用。
_SO101_DIR = os.path.dirname(os.path.abspath(__file__))
if _SO101_DIR not in sys.path:
    sys.path.insert(0, _SO101_DIR)

from so101_home_pose import (  # noqa: E402
    CONTROLLED_IDX, CONTROLLED_JOINTS, FROZEN_IDX, FROZEN_JOINTS,
    N_ACTION, N_ARM,
    RANDOMIZE_HALF_M, START_JOINT_POSE, action_to_joint_delta,
    follower_home_offset, hold_frozen, joint_delta_to_action,
    leader_home_offset, min_jerk, randomize_start_pose, tool_z, tube_tilt_deg,
)

# 启动时算一次并缓存（内部用正运动学二分求解，不写死系数）
LEADER_HOME_OFFSET = leader_home_offset(verbose=False)

# ── 常量 ────────────────────────────────────────────────────────────────
# 关节顺序与 so101_ros2_bridge 的 JOINT_NAMES 一致
JOINT_NAMES = [
    'shoulder_pan',
    'shoulder_lift',
    'elbow_flex',
    'wrist_flex',
    'wrist_roll',
    'gripper',
]
ARM_JOINTS = JOINT_NAMES[:5]          # arm_controller 只管这 5 个
GRIPPER_JOINT = JOINT_NAMES[5]

# 关节限位（取自 so101_description/urdf/so101_new_calib.urdf 的 <limit>）
JOINT_LOW = np.array([-1.920, -1.745, -1.690, -1.658, -2.744], dtype=np.float32)
JOINT_HIGH = np.array([+1.920, +1.745, +1.690, +1.658, +2.841], dtype=np.float32)
# 夹爪用弧度表示（与 bridge 的换算一致：(rad/pi)*100 → 设备单位）
GRIPPER_OPEN_RAD = 1.5      # 张开
GRIPPER_CLOSE_RAD = -0.2    # 闭合

IMG_H = IMG_W = 128
DEFAULT_HZ = 10.0

# 干预死区：leader 静止时也有传感器噪声，低于此值不视为「有意干预」
# 单位 rad（单步位移）。人手速度 ~10-50°/s，10Hz 下每步 1-5° ≈ 0.02-0.09 rad，
# 噪声约 0.1°，所以 0.05 能干净区分。需实测微调。
DEFAULT_INTERVENE_DEADBAND = 0.05

# 起始姿态实测反馈：单轮修正上限（rad），防跑飞
START_FEEDBACK_MAX = 0.35          # ≈ 20°


class SO101Env(gym.Env):
    """SO101 真机环境（ROS2 薄客户端）。"""

    def __init__(
        self,
        hz: float = DEFAULT_HZ,
        action_scale: float = 0.05,      # rad / 步（10Hz 下 ≈ 28.6°/s）
        max_episode_steps: int = 100,
        image_size: int = IMG_H,
        display_image: bool = False,
        randomize_start: bool = True,            # 起始位置随机化（pan 水平弧）
        randomize_half_m: float = RANDOMIZE_HALF_M,   # 水平弧半长（米）= 0.01 ⇒ ±1cm
        start_feedback_iters: int = 2,           # 起始姿态的实测反馈轮数（0=关闭）
        start_settle_s: float = 1.0,             # 每段斜坡后的沉降等待
        enable_keyboard: bool = True,            # 录制脚本会传 False，自己管接管状态
        node_name: str = 'so101_rl_env',
    ):
        super().__init__()
        self.hz = float(hz)
        self.action_scale = float(action_scale)
        # ★ 起始位置随机化：只转 shoulder_pan ⇒ 末端沿水平弧 ±1cm 滑动
        self.randomize_start = bool(randomize_start)
        self.randomize_half_m = float(randomize_half_m)
        self._rng = np.random.default_rng()
        # ★ 实测反馈：重力补偿是在**名义点**上标定的，换到随机点会有 1~2cm 偏差
        #   （舵机静摩擦死区同样贡献，实测可达 ~2cm）。所以量残差再补一轮 ——
        #   与 phase1_home.py 的迭代修正是同一个思路。
        self.start_feedback_iters = int(start_feedback_iters)
        self.start_feedback_tol = math.radians(0.4)
        self.start_settle_s = float(start_settle_s)
        # ★ 由录制脚本自己掌管接管状态时，必须关掉 env 自己的空格监听 ——
        #   否则按空格结束录制会**同时**把模式翻回 A（主臂力矩 ON），
        #   而回程阶段要求继续留在模式 B（主臂还要控制从臂）。
        self._enable_keyboard = bool(enable_keyboard)
        self._kin = None                 # 惰性创建（randomize_start_pose 内部会缓存）
        # ★ 重力静差补偿：读归位脚本存下来的"收敛指令 − 目标"。
        #   为什么必须有：舵机比例环无积分项，发名义角会停在名义角**下方**
        #   （实测低 3.65cm、管子歪到 8.75°）。没存档时是 0，会大声告警。
        self.home_offset = follower_home_offset()
        self.last_start_info = {'randomized': False}   # 最近一次 reset 的随机化结果
        self.max_episode_steps = int(max_episode_steps)
        self.img_size = int(image_size)
        self.display_image = display_image

        # ---------- ROS2 节点 ----------
        if not rclpy.ok():
            rclpy.init()
        self.node = Node(node_name)

        # 缓存与锁（后台线程写，主线程读）
        self._lock = threading.Lock()
        self._joint_pos: Optional[np.ndarray] = None      # (6,) rad，来自 /follower/joint_states
        self._joint_vel: Optional[np.ndarray] = None      # (6,)
        self._images: dict[str, np.ndarray] = {}          # key -> (128,128,3) uint8 RGB
        self._leader_pos: Optional[np.ndarray] = None     # (6,) rad，人类意图
        self._joint_stamp: float = 0.0

        # QoS：图像这类高频大数据用 BEST_EFFORT 更稳（丢帧没关系）
        qos_img = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
        )
        qos_js = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )

        self.create_sub = self.node.create_subscription
        self.create_sub(JointState, '/follower/joint_states', self._on_joint, qos_js)
        self.create_sub(JointState, '/leader/joint_states', self._on_leader, qos_js)
        for cam in ('cam_main', 'cam_wrist'):
            self.create_sub(
                Image, f'/follower/{cam}/image_raw',
                lambda msg, k=cam: self._on_image(msg, k), qos_img,
            )

        # 命令发布：arm 用 trajectory，gripper 用 action
        self.traj_pub = self.node.create_publisher(
            JointTrajectory, '/follower/arm_controller/joint_trajectory', 10
        )
        self.leader_cmd_pub = self.node.create_publisher(
            Float64MultiArray, '/leader/arm_command', 10
        )
        self.gripper_cli = ActionClient(
            self.node, GripperCommand, '/follower/gripper_controller/gripper_cmd'
        )

        # 力矩 service（模式切换用）
        self.leader_torque_cli = self.node.create_client(SetBool, '/leader/set_torque')
        self.follower_torque_cli = self.node.create_client(SetBool, '/follower/set_torque')

        # ---------- 后台 spin 线程 ----------
        self._stop_evt = threading.Event()
        self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._spin_thread.start()

        # ---------- 空格键监听（人类接管开关） ----------
        #
        # 已确认采用 **toggle 模式（C2）**：按一下进入接管，再按一下退出。
        #   理由：文档 3.7.1 说前 1 小时干预极其密集，[按住] 会让手很累。
        #   保留 hold 模式作为可选（SERL_SPACE_MODE=hold），以便需要"松手即回归"时使用。
        #
        # 人类干预的死区（rad，作用在受控关节的 leader 位移范数上）。
        # ★ 它只决定「这一步算不算**有意**干预」—— 即 info 里要不要出现
        #   intervene_action。**不影响** mode B 下发什么（mode B 的目标恒为
        #   foll_anchor + (leader − lead_anchor)，与死区无关）。
        #   ⚠️ 所以录制时必须把它设成 0：否则 81% 的步会记成 action=0，
        #   而手臂其实动了（实测 1~2°，最大 12.6°）→ transition 不自洽。
        self.intervene_deadband = float(DEFAULT_INTERVENE_DEADBAND)
        # 状态机在 step() 里根据 self.is_intervening 切换模式并管理力矩。
        self._space_down = False          # 物理按键当前是否按下（用于边缘检测）
        self._intervening = False         # ★ 逻辑接管状态（toggle 模式下由边缘触发翻转）
        self._space_listener = None
        self._last_mode = None            # 'policy' or 'human'
        # ★ `enable_keyboard=False` 时不装空格监听：录制脚本自己掌管接管状态
        #   （它用 d 结束录制、c 推进阶段，并按阶段显式调用 set_intervening()）。
        #   为什么要关掉：录制时按 d 结束本段，如果空格监听还在，接管状态会被
        #   空格翻转；而回程阶段**必须继续留在模式 B**（主臂还要控制从臂）。
        self.enable_keyboard = bool(enable_keyboard)
        self.space_mode = os.environ.get('SERL_SPACE_MODE', 'toggle').lower()  # 'toggle' | 'hold'
        if self.enable_keyboard:
            self._start_keyboard_listener()
        else:
            self.node.get_logger().info(
                '空格键监听已关闭（enable_keyboard=False）—— 接管状态由调用方'
                '通过 set_intervening() 显式控制')

        # 文件后备模式的边缘检测状态
        self._space_file_prev = False

        # ---------- gym spaces ----------
        f32 = np.float32
        self.observation_space = gym.spaces.Dict({
            'state': gym.spaces.Dict({
                'joint_pos': gym.spaces.Box(-np.pi, np.pi, (6,), dtype=f32),
                'joint_vel': gym.spaces.Box(-np.inf, np.inf, (6,), dtype=f32),
            }),
            'images': gym.spaces.Dict({
                'cam_main': gym.spaces.Box(0, 255, (self.img_size, self.img_size, 3), dtype=np.uint8),
                'cam_wrist': gym.spaces.Box(0, 255, (self.img_size, self.img_size, 3), dtype=np.uint8),
            }),
        })
        # ★ 动作空间 = **3 维**，只覆盖受控关节（shoulder_lift / elbow_flex /
        #   wrist_flex）。冻结的 shoulder_pan / wrist_roll / gripper
        #   **根本不进动作向量** —— 策略和人手在结构上就动不了它们。
        #   ⚠️ 夹爪不在动作空间里 ⇒ setup_mode 必须是
        #      'single-arm-fixed-gripper'（见 config.py 与 so101_home_pose.py）。
        self.action_space = gym.spaces.Box(-1.0, 1.0, (N_ACTION,), dtype=f32)

        # ---------- 内部状态 ----------
        self._step_count = 0
        self._gripper_cmd = 0.0          # 当前夹爪命令（-1..1）
        self._last_sent_gripper = None
        self._episode_return = 0.0
        self._target_pose = None      # reward 判据（set_target_joint_pose 设置）
        self._target_tol = 0.15
        self._prev_leader = None       # get_intervene_action 用
        # 人类接管的差分基准（见 _enter_mode / step 模式 B）
        self._lead_anchor = None
        self._foll_anchor = None

        # ★ 把"哪些维度不受控"打在日志里 —— 这是最容易被误解的一件事：
        #   冻结的维度指令 = 实测值，所以**策略和人手都动不了它们**。
        #   ⚠️ 冻结 shoulder_pan 后，末端被限制在一个竖直平面内（2 个平移自由度）。
        self.node.get_logger().info(
            f'动作空间：跟随 {list(CONTROLLED_JOINTS)}；'
            f'冻结 {list(FROZEN_JOINTS)}（指令=实测值，永不改变）'
        )

    # ================= 键盘（接管开关） =================
    #
    # WSL2 无头环境实测结论（2026-09）：
    #   · pynput xorg 后端  → 需要 X server；本机 DISPLAY=127.0.0.1:0.0 连不上  ❌
    #   · pynput uinput 后端 → 该版本缺 pynput.mouse._uinput                    ❌
    #   · /dev/input 不存在  → 读不到物理输入设备                                ❌
    #   · termios + stdin    → ✅ 可用（训练时操作者就坐在终端前）
    # 所以主方案是 stdin，pynput 仅作有显示环境下的备选。
    def _start_keyboard_listener(self):
        """监听空格键。已确认采用 **toggle 模式**：按一下接管，再按一下退出。

        （SERL_SPACE_MODE=hold 可切回「按住生效、松手即回归」。）

        优先级：
          1. stdin + termios（无头环境唯一可用；本机实际用它）
          2. pynput（有 X server 时可用；本机不可用）
        都不可用时打印警告并禁用接管（训练仍可跑，只是没有人工介入）。
        """
        self._space_listener = None
        self._space_down = False
        self._kb_mode = None

        mode = self._try_stdin_keyboard()
        if mode:
            self._kb_mode = mode
            if mode == 'file':
                self.node.get_logger().info(
                    f'空格键接管使用**文件模拟**（stdin 非 TTY）：{os.environ.get("SERL_SPACE_FILE")}\n'
                    '   文件存在 == 空格按下。仅用于自动化测试；真机训练请用交互式终端。'
                )
            else:
                _hint = ('按一下空格=进入接管，再按一下=退出'
                         if self.space_mode == 'toggle' else
                         '按住空格=人类接管，松开=回到 policy')
                self.node.get_logger().info(
                    f'空格键接管已启用（stdin/termios，模式={self.space_mode}）—— {_hint}'
                )
            return

        if self._try_pynput_keyboard():
            self._kb_mode = 'pynput'
            self.node.get_logger().info(
                f'空格键接管已启用（pynput，模式={self.space_mode}）'
            )
            return

        self.node.get_logger().warn(
            '⚠️ 键盘监听不可用（stdin 不是 TTY，且无可用 X server）—— 人类接管被禁用。\n'
            '   启用方法：在**交互式终端**里直接运行训练脚本（不要用管道或重定向）；\n'
            '   或设 SERL_SPACE_FILE=<path>，脚本会以「该文件存在」代表「空格按下」（自动化测试用）。'
        )

    def _on_space_press(self):
        """★ toggle 核心：每次「按下」事件翻转接管状态。

        toggle 模式：只在按下沿翻转，因此"按一下进入、再按一下退出"。
        hold  模式：按下时置 True（由 _on_space_release 置 False）。

        注意：stdin/pynput 后端拿不到 key release（stdin 只读字符流，
        终端不发 release 事件），所以 toggle 是这两个后端的天然选择；
        hold 模式在 stdin 后端下无法可靠工作（会一直视为按下），
        因此 stdin 后端建议配 toggle（默认）。
        """
        if self.space_mode == 'toggle':
            self._intervening = not self._intervening
            self.node.get_logger().info(
                f'[空格] 接管={"ON" if self._intervening else "OFF"}（toggle）'
            )
        else:
            if not self._space_down:
                self._intervening = True
        self._space_down = True

    def _on_space_release(self):
        """hold 模式下的松开事件（仅 pynput 等能拿到 release 的后端有效）。"""
        self._space_down = False
        if self.space_mode == 'hold':
            self._intervening = False

    def _try_stdin_keyboard(self):
        """用 termios 把 stdin 设为 cbreak，后台线程 select 检测空格。

        Returns:
            'stdin'  —— 成功使用终端 stdin
            'file'   —— stdin 非 TTY，退化为 SERL_SPACE_FILE 文件模拟
            None     —— 都不可用

        ⚠️ 会修改终端模式，close() 时必须恢复（否则终端会变得很难用）。
        """
        import sys
        try:
            import select
            import termios
            import tty

            if not sys.stdin.isatty():
                # 非 TTY（管道/重定向/自动化）→ 用文件模拟
                return 'file' if self._try_space_file() else None

            fd = sys.stdin.fileno()
            self._stdin_fd = fd
            self._stdin_old_attrs = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        except Exception:
            return False

        def _loop():
            while not self._stop_evt.is_set():
                try:
                    r, _, _ = select.select([self._stdin_fd], [], [], 0.05)
                    if not r:
                        continue
                    ch = sys.stdin.read(1)
                    if ch == ' ':
                        self._on_space_press()      # ← 每次按下触发一次（toggle）
                    elif ch == '\n':
                        pass                        # 回车忽略，避免误触
                    elif ch == '\x03':          # Ctrl-C
                        self._intervening = False
                        self._stop_evt.set()
                        break
                except Exception:
                    time.sleep(0.1)

        self._stdin_thread = threading.Thread(target=_loop, daemon=True)
        self._stdin_thread.start()
        return 'stdin' 

    def _try_space_file(self) -> bool:
        """测试后备：轮询 SERL_SPACE_FILE 指定的文件是否存在（存在=空格按下）。

        仅当 stdin 不是 TTY 时启用，这样自动化测试也能验证接管逻辑。
        """
        path = os.environ.get('SERL_SPACE_FILE')
        if not path:
            return False
        self._space_file = path

        def _loop():
            while not self._stop_evt.is_set():
                try:
                    cur = os.path.exists(self._space_file)
                    # 只在「文件从无到有」的边缘触发一次，等价于按键按下
                    if cur and not self._space_file_prev:
                        self._on_space_press()
                    elif not cur and self._space_file_prev:
                        self._on_space_release()
                    self._space_file_prev = cur
                except Exception:
                    pass
                time.sleep(0.05)

        self._space_thread = threading.Thread(target=_loop, daemon=True)
        self._space_thread.start()
        self.node.get_logger().info(f'空格键接管使用文件模拟：{path}')
        return True

    def _try_pynput_keyboard(self) -> bool:
        try:
            from pynput import keyboard

            def _kp(k):
                if k == keyboard.Key.space:
                    self._on_space_press()

            def _kr(k):
                if k == keyboard.Key.space:
                    self._on_space_release()

            self._space_listener = keyboard.Listener(on_press=_kp, on_release=_kr)
            self._space_listener.daemon = True
            self._space_listener.start()
            return True
        except Exception:
            return False

    def set_intervening(self, on: bool):
        """Explicitly set the intervention state (used by the recording script).

        Only meaningful when `enable_keyboard=False`, i.e. when the caller owns
        the mode state machine instead of the env's spacebar toggle.
        """
        self._intervening = bool(on)

    @property
    def is_intervening(self) -> bool:
        """人类是否正在接管。

        toggle 模式：按一下空格置 True，再按一下置 False。
        hold   模式：按住为 True，松开为 False。
        """
        return bool(self._intervening)

    def _enter_mode(self, mode: str):
        """切换控制模式并管理力矩（只在模式变化时执行一次）。

        ★ 进入人类接管时**必须**记下差分基准 (follower 实测, leader 实测) —— 见 step()
          里模式 B 的说明。漏掉它会让从臂在接管瞬间跳回主臂所在姿态。
        """
        if mode == self._last_mode:
            return
        if mode == 'human':
            # 人类接管：leader 必须松开，否则人手和电机较劲（抖振/发热）
            ok = self.set_leader_torque(False)
            # ★ 差分基准：此刻从臂在哪、主臂在哪。之后从臂 = 基准 + (主臂 − 主臂基准)。
            if self.leader_pos is not None and self.joint_pos is not None:
                self._lead_anchor = np.asarray(self.leader_pos, float)[:N_ARM].copy()
                self._foll_anchor = np.asarray(self.joint_pos, float)[:N_ARM].copy()
                gap = float(np.abs(self._foll_anchor - self._lead_anchor).max())
                self.node.get_logger().info(
                    f'>>> 人类接管（leader 力矩 OFF: {ok}）差分基准已记录，'
                    f'两臂当前相差 {math.degrees(gap):.1f}°')
            else:
                self._lead_anchor = self._foll_anchor = None
                self.node.get_logger().warn(
                    f'>>> 人类接管（leader 力矩 OFF: {ok}）但拿不到关节状态，'
                    f'差分基准缺失，本步骤将退化为保持不动')
        else:
            # 回到 policy：leader 恢复力矩，重新跟随 follower
            self._lead_anchor = self._foll_anchor = None
            ok = self.set_leader_torque(True)
            self.node.get_logger().info(f'<<< 回到 policy（leader 力矩 ON: {ok}）')
        self._last_mode = mode

    # ================= 后台线程 =================
    def _spin_loop(self):
        while not self._stop_evt.is_set():
            try:
                rclpy.spin_once(self.node, timeout_sec=0.05)
            except Exception:
                break

    # ================= 回调（写缓存） =================
    def _reorder(self, msg: JointState) -> Optional[np.ndarray]:
        """按 JOINT_NAMES 顺序取 position；缺 name 时退化为原顺序。"""
        if not msg.name:
            return np.asarray(msg.position, dtype=np.float32)
        try:
            return np.asarray(
                [msg.position[msg.name.index(j)] for j in JOINT_NAMES], dtype=np.float32
            )
        except (ValueError, IndexError):
            return np.asarray(msg.position, dtype=np.float32)

    def _on_joint(self, msg: JointState):
        pos = self._reorder(msg)
        if pos.shape != (6,):
            return
        vel = None
        if msg.velocity and len(msg.velocity) >= 6:
            vel = np.asarray(msg.velocity, dtype=np.float32)
        with self._lock:
            self._joint_pos = pos
            if vel is not None:
                self._joint_vel = vel
            self._joint_stamp = time.time()

    def _on_leader(self, msg: JointState):
        pos = self._reorder(msg)
        if pos.shape == (6,):
            with self._lock:
                self._leader_pos = pos

    def _on_image(self, msg: Image, key: str):
        """把 ROS2 Image 转成 (H, W, 3) uint8。

        ⚠️ 颜色顺序说明：
          usb_cam 的配置写的是 pixel_format="mjpeg2rgb"，理论上输出 rgb8，
          但不同版本行为不一致（有的仍发 bgr8）。这里**按 msg.encoding 判断**，
          而不是硬假设，避免"图看着正常但通道反了"这种隐蔽 bug
          （对 reward classifier 影响很大：橙子/盘子颜色会失真）。

          内部统一存 **RGB**（HIL-SERL 的 RSCapture 也是 RGB）。

        分辨率：相机可能不是 640x480（全局相机我们配了 640x480，
        腕部是 424x240），所以一律 resize 到 img_size。
        """
        try:
            if msg.encoding not in ('rgb8', 'bgr8', '8UC3', ''):
                # 其它编码（如 yuv422）暂不支持，直接丢弃该帧
                return
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            img = arr.reshape((msg.height, msg.width, -1))[:, :, :3]
            if img.shape[0] != self.img_size or img.shape[1] != self.img_size:
                img = cv2.resize(img, (self.img_size, self.img_size),
                                 interpolation=cv2.INTER_AREA)
            # 统一成 RGB
            if msg.encoding == 'bgr8':
                img = img[:, :, ::-1].copy()
            else:
                img = img.copy()
            # 记录一次实际编码，便于排查
            if getattr(self, '_last_encoding', {}).get(key) != msg.encoding:
                self._last_encoding = getattr(self, '_last_encoding', {})
                self._last_encoding[key] = msg.encoding
                self.node.get_logger().info(
                    f'[{key}] 图像编码 = {msg.encoding!r}，'
                    f'{msg.width}x{msg.height} -> {self.img_size}x{self.img_size}'
                )
        except Exception:
            return
        with self._lock:
            self._images[key] = img

    # ================= 读取缓存 =================
    @property
    def joint_pos(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._joint_pos is None else self._joint_pos.copy()

    @property
    def leader_pos(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._leader_pos is None else self._leader_pos.copy()

    def _get_obs(self) -> dict:
        with self._lock:
            pos = self._joint_pos
            vel = self._joint_vel
            imgs = dict(self._images)
        if pos is None:
            pos = np.zeros(6, dtype=np.float32)
        if vel is None:
            vel = np.zeros(6, dtype=np.float32)
        images = {}
        for cam in ('cam_main', 'cam_wrist'):
            images[cam] = imgs.get(cam, np.zeros((self.img_size, self.img_size, 3), np.uint8))
        if self.display_image:
            try:
                show = np.concatenate([images['cam_main'], images['cam_wrist']], axis=1)
                cv2.imshow('so101_env', show[:, :, ::-1])   # 显示要 BGR
                cv2.waitKey(1)
            except Exception:
                pass
        return {
            'state': {'joint_pos': pos.astype(np.float32), 'joint_vel': vel.astype(np.float32)},
            'images': images,
        }

    # ================= 发命令 =================
    def _publish_arm(self, target5: np.ndarray):
        msg = JointTrajectory()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.joint_names = list(ARM_JOINTS)
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in target5]
        pt.time_from_start.sec = 0
        pt.time_from_start.nanosec = 3_000_000     # 3ms，与 teleop 组件一致
        msg.points = [pt]
        self.traj_pub.publish(msg)

    def _send_gripper(self, cmd: float):
        """cmd ∈ [-1,1]：>0 张开，<0 闭合。带去抖动（变化小就不发）。"""
        cmd = float(np.clip(cmd, -1.0, 1.0))
        if abs(cmd - self._gripper_cmd) < 0.1 and self._last_sent_gripper is not None:
            return
        self._gripper_cmd = cmd
        target = GRIPPER_OPEN_RAD if cmd >= 0 else GRIPPER_CLOSE_RAD
        if not self.gripper_cli.server_is_ready():
            self.gripper_cli.wait_for_server(timeout_sec=0.05)
        if not self.gripper_cli.server_is_ready():
            return
        goal = GripperCommand.Goal()
        goal.command.position = float(target)
        goal.command.max_effort = 10.0
        self.gripper_cli.send_goal_async(goal)
        self._last_sent_gripper = cmd

    # ================= 力矩 =================
    def _call_bool(self, cli, value: bool, timeout: float = 2.0) -> bool:
        if not cli.service_is_ready():
            cli.wait_for_service(timeout_sec=timeout)
        if not cli.service_is_ready():
            return False
        req = SetBool.Request()
        req.data = bool(value)
        fut = cli.call_async(req)
        t0 = time.time()
        while not fut.done() and time.time() - t0 < timeout:
            time.sleep(0.02)
        return bool(fut.done() and fut.result() is not None and fut.result().success)

    def set_leader_torque(self, on: bool) -> bool:
        return self._call_bool(self.leader_torque_cli, on)

    def set_follower_torque(self, on: bool) -> bool:
        return self._call_bool(self.follower_torque_cli, on)

    # ================= reward（占位：可配置目标关节位姿） =================
    def set_target_joint_pose(self, pose: Optional[np.ndarray], tol: float = 0.15):
        """设置「成功」判据：所有关节与目标位姿的差都 < tol（rad）。

        pose=None 表示不判成功（reward 恒 0）。
        这是 B0.5 的规则版 reward，后续可替换为 reward classifier。
        """
        self._target_pose = None if pose is None else np.asarray(pose, dtype=np.float32)
        self._target_tol = float(tol)

    def compute_reward(self, obs: dict) -> int:
        tgt = getattr(self, '_target_pose', None)
        if tgt is None:
            return 0
        pos = obs['state']['joint_pos']
        # 只比前 5 个关节（夹爪由任务定义决定，先不参与）
        return int(np.all(np.abs(pos[:5] - tgt[:5]) < getattr(self, '_target_tol', 0.15)))

    # ================= gym API =================
    def _publish_arm_ramp(self, q_to, n: int = 25, v_max_deg: float = 30.0,
                          d_min: float = 0.6, d_max: float = 3.0, hold_idx=None):
        """用一条**多点 min-jerk 轨迹**把从臂平滑送到 q_to，返回轨迹时长（秒）。

        为什么不能只发一个 `time_from_start = 3ms` 的点：
            `arm_controller`（JointTrajectoryController）在
            `so101_controllers.yaml` 里**没有配任何 velocity / acceleration 约束**，
            所以 3ms 的点就等于"立刻到位" —— 从臂会**突然弹过去**。
            归位时夹爪里还夹着试管，冲击会直接传到试管和舵机上。
            （2026-09-29 排查确认：该 yaml 里只有 gains / update_rate /
             time_scaling，没有任何 constraints。）

        为什么用**多点轨迹**而不是循环发单点：
            JTC 自己会按每个点的 `time_from_start` 插值，我们只要睡够时长；
            循环发单点反而会因为发布抖动变成一串小跳。

        `hold_idx` 指定**哪些维度锁在实测值**（默认 FROZEN_IDX）。
        ⚠️ `reset()` 会传 `hold_idx=(4,)`（只锁 wrist_roll）：因为**起点姿态本身
           包含 shoulder_pan**，如果 pan 被推动过、而 reset 又把它锁在实测值，
           那"起点"就不成立了 —— 平面被转掉，插孔会够不着。

        时长按 min-jerk 峰值速度反推（峰值 = 1.875 × 位移 / 时长）。
        """
        q_to = np.asarray(q_to, dtype=float).reshape(-1)[:N_ARM]
        hold = list(FROZEN_IDX if hold_idx is None else hold_idx)
        cur = self.joint_pos
        q_from = (np.asarray(cur, dtype=float).reshape(-1)[:N_ARM].copy()
                  if cur is not None else q_to.copy())
        for i in hold:
            if i < N_ARM:
                q_from[i] = q_from[i]          # 起点已在实测值上，这里只是显式化
                q_to[i] = q_from[i]

        disp = float(np.abs(q_to - q_from).max())
        if disp < 1e-6:
            self._publish_arm(q_to)
            return 0.0          # 已经在那儿了，不用动
        dur = float(np.clip(1.875 * math.degrees(disp) / v_max_deg, d_min, d_max))

        msg = JointTrajectory()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.joint_names = list(ARM_JOINTS)
        for k in range(n + 1):
            t = dur * k / n
            pt = JointTrajectoryPoint()
            pt.positions = [float(v) for v in (q_from + (q_to - q_from) * min_jerk(k / n))]
            pt.time_from_start.sec = int(t)
            pt.time_from_start.nanosec = int(round((t - int(t)) * 1e9))
            msg.points.append(pt)
        self.traj_pub.publish(msg)
        return dur

    def reset(self, *, seed=None, options=None, wait_for_data: float = 5.0):
        """回到（随机化的）初始位姿并返回第一帧观测。

        起点 = Phase 0 验证过的 `START_JOINT_POSE`，再只转 shoulder_pan 做
        水平弧随机化（见 so101_home_pose.randomize_start_pose）。
        """
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._episode_return = 0.0
        self._gripper_cmd = 0.0
        self._last_sent_gripper = None

        # 等拿到第一帧关节数据（避免观测全 0）
        t0 = time.time()
        while self.joint_pos is None and time.time() - t0 < wait_for_data:
            time.sleep(0.05)

        # ★★ 起点：Phase 0 验证过的姿态 + shoulder_pan 水平弧随机化 ★★
        #   不再用 INIT_POSE —— 那是个**占位折叠姿态**（pan=0），既不是验证过的
        #   起点、又会把 pan 摇 19°。
        #   ⚠️ 随机化只有 2 个方向：shoulder_pan 冻结 ⇒ 末端被锁在一个竖直平面内，
        #      做不出水平面内的二维方形随机化（用户 2026-09-29 决定接受）。
        q5 = np.asarray(START_JOINT_POSE, dtype=float)[:N_ARM].copy()
        # 起点姿态 = START_JOINT_POSE（5 个臂关节全部），受控关节由随机化 IK 覆盖。
        # ⚠️ 2026-09-29 起 pan 已解冻（在 CONTROLLED_IDX 里），所以不再需要
        #    "把 pan 拉回名义值"那种补丁 —— IK 自己会给出合适的 pan。
        #    wrist_roll 仍锁实测：管子轴对称，它的值对任务没有意义，不动更稳。
        if self.joint_pos is not None:
            q5[4] = float(self.joint_pos[4])
        start_info = {'randomized': False}
        if self.randomize_start:
            q_rand, info = randomize_start_pose(
                self._rng, half_m=self.randomize_half_m,
                q_cur=self.joint_pos, kin=self._kin)
            if info.get('ok'):
                q5[list(CONTROLLED_IDX)] = np.asarray(q_rand)[list(CONTROLLED_IDX)]
                start_info = {'randomized': True, **{
                    k: info[k] for k in ('dx_m', 'dy_m', 'dz_m', 'err_mm',
                                         'tube_angle_deg', 'd_tilt_deg',
                                         'margin_rad', 'iters')
                }}
            else:
                # ★ 绝不把没通过复核的解下发 —— 退回名义起点，并把原因打出来。
                #   不是"以防万一"：IK 在框边界上确实会差一点，
                #   下发一个姿态错的目标比"这次不随机"糟糕得多。
                self.node.get_logger().warn(
                    f'起始位置随机化未通过复核，本次退回名义起点：{info.get("reason")}')
                start_info = {'randomized': False, 'reason': info.get('reason')}
        self.last_start_info = start_info
        if start_info['randomized']:
            self.node.get_logger().info(
                f'起始位置随机化：dx {start_info["dx_m"]*100:+.2f}cm / '
                f'dy {start_info["dy_m"]*100:+.2f}cm / '
                f'dz {start_info["dz_m"]*100:+.2f}cm，'
                f'管子距竖直 {start_info["tube_angle_deg"]:.3f}°，'
                f'限位余量 {start_info["margin_rad"]:.2f}rad')

        # ★ 重力静差补偿：q5 是希望手臂**实际**停在的姿态，而舵机是比例环、
        #   发什么就停在什么下方几度。所以下发的**指令**要加上归位脚本收敛出来的
        #   那个偏置。冻结维度的偏置无所谓 —— _publish_arm_ramp 里会被实测值覆盖。
        q_cmd = q5 + self.home_offset

        # ★ 不发夹爪指令：夹爪是**冻结**的，一直夹着试管。
        #   踩坑（2026-09-29 排查发现）：原代码这里发 `_send_gripper(-1.0)`，
        #   而 GRIPPER_CLOSE_RAD = −0.2 rad，会把夹爪从保存的 +0.325 rad
        #   **压过去 0.5 rad** —— 每个 episode 都挤一次试管。既然夹爪不参与控制，
        #   就一个指令都不该发（它的位置由 ros2_control 的夹爪控制器保持）。
        #   将来若要放开夹爪，这里恢复 _send_gripper 即可。

        # 平滑走过去（不是弹过去）；冻结维度在 _publish_arm_ramp 里已锁在实测值
        dur = self._publish_arm_ramp(q_cmd, hold_idx=(4,))   # 只锁 wrist_roll
        time.sleep(dur + self.start_settle_s)

        # ★★ 实测反馈：把"实际停在哪"量回来、把残差补进指令，再走一小段 ★★
        #   为什么不能只靠固定的重力补偿：那个偏置是在**名义点**上标定的，
        #   换到随机点后重力力矩变了、静摩擦死区也变了，实测会偏高 1~2cm
        #   （实测：请求竖直 +2.44cm → 实际 +4.42cm）。这与 phase1_home.py
        #   的迭代修正是同一个思路，只是这里只跑 1~2 轮。
        n_fb = 0
        for _ in range(max(0, self.start_feedback_iters)):
            cur = self.joint_pos
            if cur is None:
                break
            q_act = np.asarray(cur, dtype=float).reshape(-1)[:N_ARM]
            e = q5 - q_act
            for i in FROZEN_IDX:                     # 冻结维度不修正
                if i < N_ARM:
                    e[i] = 0.0
            if float(np.abs(e).max()) < self.start_feedback_tol:
                break
            e = np.clip(e, -START_FEEDBACK_MAX, START_FEEDBACK_MAX)
            q_cmd = q_cmd + e
            d2 = self._publish_arm_ramp(q_cmd, hold_idx=(4,))
            time.sleep(d2 + self.start_settle_s)
            n_fb += 1
        start_info['feedback_iters'] = n_fb
        # ★ 报出"实际停在哪 vs 目标"的残差 —— phase1_home.py 一直有这个数字，
        #   reset 之前没有，于是"复位不如归位脚本准"只能靠感觉。现在能量了。
        cur = self.joint_pos
        if cur is not None:
            q_act = np.asarray(cur, dtype=float).reshape(-1)[:N_ARM]
            start_info['resid_deg'] = float(
                math.degrees(np.abs(q5 - q_act)[list(CONTROLLED_IDX)].max()))

        # ★ 复核：量**实际**停在哪、管子多歪。这是"重力补偿到底对不对"的唯一
        #   现场证据。补偿为 0（没跑归位脚本）时管子会歪到 ~8.8°，这里会告警。
        cur = self.joint_pos
        if cur is not None:
            q_act = np.asarray(cur, dtype=float).reshape(-1)[:N_ARM]
            start_info['act_tilt_deg'] = tube_tilt_deg(q_act)
            start_info['act_z_cm'] = tool_z(q_act) * 100.0
            if start_info['act_tilt_deg'] > 3.0:
                self.node.get_logger().warn(
                    f'起始姿态管子歪了 {start_info["act_tilt_deg"]:.2f}°'
                    f'（名义起点是 1.66°），末端 z={start_info["act_z_cm"]:.1f}cm —— '
                    f'多半是**重力静差补偿没生效**。先跑一次 phase1_home.py 生成'
                    f'归位指令存档，再重开训练。')
            else:
                self.node.get_logger().info(
                    f'起始姿态：管子距竖直 {start_info["act_tilt_deg"]:.2f}°，'
                    f'末端 z={start_info["act_z_cm"]:.1f}cm')
        return self._get_obs(), {'succeed': False, 'start': start_info}

    def set_intervene_deadband(self, rad: float):
        """设置干预判定的死区（rad）。录制前应传 0.0。

        为什么录制要 0：死区只影响「info 里报不报 intervene_action」，
        而 mode B 的从臂命令**不受它影响**。死区 0.05 rad（2.86°）比 10Hz 下
        典型的人类单步位移还大，于是慢速操作时：手臂动了 1~2°，
        录下来的 action 却是 0 —— `(obs, action, next_obs)` 不再自洽，
        RLPD 的 critic 会学到一个"零动作也能让手臂动"的假世界。
        实测首批 52 段：**81.1% 的步 action=0，其中大半手臂确实动了**。
        """
        self.intervene_deadband = float(rad)
        return self.intervene_deadband

    def get_intervene_action(self, deadband: Optional[float] = None) -> Optional[np.ndarray]:
        """若人类正在接管，返回与 policy 同语义的「专家动作」（关节增量，∈[-1,1]^3）。

        返回 None 表示本步没有人类意图。

        ★ 维度 = 3，与 `self.action_space` 完全一致（只含受控关节：
          shoulder_lift / elbow_flex / wrist_flex）。冻结关节不在这里面。

        语义必须与 policy 动作严格一致（都是关节增量），否则 train_rlpd 的
        `actions = info.pop("intervene_action")` 会把两种量纲的动作混进同一个
        buffer，critic 学出来就是坏的。

        Q1 决策：以 leader 的「增量」判定，不是绝对位置。
          原版 SpacemouseIntervention 用 `norm(expert_a) > 0.001`，
          但 SpaceMouse 静止时输出≈0，而 leader 关节角是绝对位置（永不为 0），
          照抄会导致「永远在干预」。
        """
        if not self.is_intervening:
            self._prev_leader = self.leader_pos
            return None

        cur = self.leader_pos
        if cur is None:
            return None
        prev = getattr(self, '_prev_leader', None)
        self._prev_leader = cur
        if prev is None:
            return None

        # ★ 死区默认值可用 set_intervene_deadband() 覆盖：
        #   录制时必须设成 0 —— 见 set_intervene_deadband 的说明。
        db = (getattr(self, 'intervene_deadband', DEFAULT_INTERVENE_DEADBAND)
              if deadband is None else float(deadband))
        delta = cur - prev

        # 专家关节增量 = leader 位移 / action_scale，使其与 policy 动作同量纲
        # ★ 维度是 **3**（只取受控关节），必须与 `self.action_space`、
        #   以及 policy 输出的动作**逐维对齐** —— train_rlpd 的
        #   `actions = info.pop("intervene_action")` 会把两者塞进同一个 buffer，
        #   形状或语义对不上，critic 学出来就是坏的。
        #   顺带消掉原先把 leader 的 gripper **弧度**当 [0,100] 用
        #   （`(cur[5]-50)/50`，恒为 ≈ −1）那个潜在 bug 的影响。
        expert = np.clip(joint_delta_to_action(delta[:N_ARM], self.action_scale),
                         -1.0, 1.0).astype(np.float32)

        # 死区：leader 静止时也有传感器噪声，低于阈值不算「有意干预」
        # ★ 只在**受控关节**上判：冻结关节（pan / wrist_roll / gripper）即使被人手
        #   推动，也产生不了任何动作（expert 恒 0），不该算成"有意干预" ——
        #   否则会往 buffer 里塞进一批"专家动作全 0"的样本，把 critic 带偏。
        if np.linalg.norm(delta[list(CONTROLLED_IDX)]) < db:
            return None
        return expert

    def step(self, action: np.ndarray):
        # ---- 模式切换（含力矩管理）----
        self._enter_mode('human' if self.is_intervening else 'policy')

        cur = self.joint_pos
        if cur is None:
            cur = np.zeros(6, dtype=np.float32)

        # ---- 目标位姿：人类接管时 follower 跟随 leader，否则跟随 policy ----
        # ★ 动作是 **3 维**，只作用在受控关节上（action_to_joint_delta 会把
        #   它散射到 5 关节增量里，冻结维恒 0）。
        # ★ 夹爪不在动作空间里：本任务夹爪一直夹着试管，reset() 闭合到位后
        #   step() **不再发任何夹爪指令**。（将来若要放开夹爪，需同时把
        #   'gripper' 移出 FROZEN_JOINTS 并在这里恢复 _send_gripper。）
        if self.is_intervening:
            # 模式 B：follower 跟随 leader 的**变化量**（差分映射）
            #
            #   target5 = 接管瞬间的从臂位置 + (主臂当前位置 − 接管瞬间的主臂位置)
            #
            # ★★ 为什么必须是差分，不能用"主臂绝对角"（2026-09-29 改）★★
            #   原先写的是 `target5 = leader[:5] − LEADER_HOME_OFFSET`（绝对）。
            #   那有两个后果，都会**静默地毁掉数据**：
            #     ① reset() 刚把从臂随机化到某个起点，人手一接管，从臂立刻跳回
            #        主臂所在的姿态 —— 随机化被当场抹掉（练习到的起点全都一样）。
            #     ② 更隐蔽：录下来的 `intervene_action` 是"主臂位移 / action_scale"，
            #        而 env 实际下发的是主臂的**绝对角** —— 两者根本不是一回事。
            #        RLPD 的 transition 要求 (obs, action, next_obs) 自洽
            #        （"在 obs 施加 action 得到 next_obs"），绝对映射下这个等式不成立，
            #        critic 学到的东西是自相矛盾的。
            #   改成差分后：从臂位移 == 主臂位移 == action × action_scale ✓ 自洽。
            #   顺带也不需要再显式减 LEADER_HOME_OFFSET 了 —— 主臂的家偏置
            #   在 `主臂当前位置 − 主臂基准` 里自动抵消。
            led = self.leader_pos
            if led is not None and self._lead_anchor is not None:
                target5 = (self._foll_anchor + (np.asarray(led, float)[:N_ARM]
                                            - self._lead_anchor)
                       + self.home_offset)
            elif led is not None:
                # 基准缺失（拿不到状态就接管了）—— 安全退化：保持不动
                target5 = cur[:5].copy() + self.home_offset
            else:
                target5 = (cur[:5] + action_to_joint_delta(action, self.action_scale)
                           + self.home_offset)
        else:
            # 模式 A：policy 控制。关节增量以**实测位置**为基准（避免累积漂移）
            #
            # ★★ 必须再加 `home_offset`（重力静差补偿）★★
            #   不加会怎样（2026-09-29 真机现象，用户报的"复位不准 / 接管时从臂
            #   会变一下"）：
            #     模式 A 的指令 = 实测位置 → 舵机的比例误差为 0 → **没有顶重力的力**
            #     → 手臂一路缓慢下垂。而接管瞬间模式 B 的指令是**冻结住的**
            #     `foll_anchor + home_offset`，比已经垂下去的位置高 → 从臂"弹回去一下"。
            #   加上这个偏置后：模式 A 停在指令附近（不再漂），接管时
            #   `foll_anchor + home_offset` 与模式 A 的指令**恰好连续** → 零跳变。
            #   这正是用户提的"把固定时候的那个作为一个残差项"。
            target5 = (cur[:5] + action_to_joint_delta(action, self.action_scale)
                       + self.home_offset)

        target5 = np.clip(target5, JOINT_LOW, JOINT_HIGH)

        # ★ 冻结维度兜底：指令 = **实测值**，也就是"根本不命令它"。
        #   模式 A 下 action_to_joint_delta 已经保证冻结维为 0，这一步是冗余的；
        #   但模式 B 的目标来自主臂（含 pan/wrist_roll），**必须**过这一道。
        #   用实测值而不是固定目标角：零回复力、永不跳变。
        target5 = hold_frozen(target5, cur)
        self._publish_arm(target5)

        # 模式 A 下让 leader 跟随 follower（有手感反馈）
        if not self.is_intervening:
            self._drive_leader(cur)

        # 维持控制周期
        time.sleep(1.0 / self.hz)

        obs = self._get_obs()
        reward = self.compute_reward(obs)
        self._step_count += 1
        self._episode_return += reward
        done = self._step_count >= self.max_episode_steps or reward == 1

        info = {'succeed': bool(reward), 'intervening': self.is_intervening}
        expert = self.get_intervene_action(self.intervene_deadband)
        if expert is not None:
            info['intervene_action'] = expert      # ← RLPD 靠这个字段
        return obs, int(reward), done, False, info

    def _drive_leader(self, follower_pos: np.ndarray):
        """模式 A：把 follower 当前位置发给 leader，让它跟随。

        ★ 必须加上 LEADER_HOME_OFFSET（主臂的家比从臂高）。漏掉的话主臂会被
          拽到从臂的姿态上 → 贴住自己那张更高的桌面 → 堵转 → 过热保护。
        注意 `/leader/arm_command` 要求**正好 6 个数**，所以 gripper 原样带上。
        """
        q = np.asarray(follower_pos, dtype=float).reshape(-1).copy()
        q[:N_ARM] = q[:N_ARM] + LEADER_HOME_OFFSET
        m = Float64MultiArray()
        m.data = [float(v) for v in q]
        self.leader_cmd_pub.publish(m)

    # ================= 清理 =================
    def close(self):
        # ★ 必须恢复终端模式，否则 shell 会一直处于 cbreak（输入变得很怪）
        if getattr(self, '_kb_mode', None) == 'stdin' and hasattr(self, '_stdin_old_attrs'):
            try:
                import termios
                termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._stdin_old_attrs)
            except Exception:
                pass
        if getattr(self, '_space_listener', None) is not None:
            try:
                self._space_listener.stop()
            except Exception:
                pass
        self._stop_evt.set()
        try:
            self._spin_thread.join(timeout=2.0)
        except Exception:
            pass
        try:
            self.node.destroy_node()
        except Exception:
            pass
        if self.display_image:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


# 初始位姿：取自 so101_description/config/initial_positions.yaml
# ⚠️ 已废弃、**不要再用**：这是一个占位折叠姿态（pan=0，不是 Phase 0 验证过的起点）。
#   2026-09-29 起 reset() 改用 START_JOINT_POSE + 竖直平面内随机化。
#   保留在此只为让 `git diff` 看得清"改了什么"，避免有人误以为它还是初值。
INIT_POSE = np.array([0.0, -1.75, 1.58, 1.0, 0.25, 0.0], dtype=np.float32)  # UNUSED
