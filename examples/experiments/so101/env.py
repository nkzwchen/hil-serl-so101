"""SO101 的 gym 环境（HIL-SERL 用）。

设计要点（已与使用者确认）：
  · 命令路径：直接发 /follower/arm_controller/joint_trajectory + gripper action
              （env 是 follower 的唯一控制源，避免与 teleop 组件抢话题）
  · 观测源  ：/follower/joint_states（ros2_control 广播，io.yaml 同款）
              + /follower/cam_{main,wrist}/image_raw
  · 动作空间：pan / shoulder_lift / elbow_flex / wrist_flex 的相对角度，4 维。
              每关节最多 0.02rad/步；沿上一条已接受命令累积，保留归位补偿。
              wrist_roll / gripper 冻结；动作阶段不运行 IK 或末端姿态约束。
  · 不用 RelativeFrame：动作直接表达为关节角增量。
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

from latest_frame import LatestFrameWorker
from camera_observation import CAMERA_KEYS, decode_rgb, crop_roi
from joint_control import CommandHold, bounded_joint_delta, ACTION_SEMANTICS, DEFAULT_STEP_RAD, execute_joint_request

from so101_home_pose import (  # noqa: E402
    CONTROLLED_IDX, CONTROLLED_JOINTS, FROZEN_IDX, FROZEN_JOINTS,
    N_ACTION, N_ARM,
    RANDOMIZE_HALF_M, START_JOINT_POSE, action_to_joint_delta,
    follower_home_offset, hold_frozen, joint_delta_to_action,
    leader_home_offset, min_jerk, randomize_joint_start_pose, tool_z, tube_tilt_deg,
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
START_FEEDBACK_MAX = math.radians(3.0)  # per-pass correction cap
START_ACCEPTANCE_TOL = math.radians(3.0)  # final settled joint residual limit


class SO101Env(gym.Env):
    """SO101 真机环境（ROS2 薄客户端）。"""

    def __init__(
        self,
        hz: float = DEFAULT_HZ,
        action_scale: float = DEFAULT_STEP_RAD,      # radians per joint per step
        max_episode_steps: int = 100,
        image_size: int = IMG_H,
        display_image: bool = False,
        randomize_start: bool = True,            # pan/lift/elbow 关节随机化
        randomize_half_m: float = RANDOMIZE_HALF_M,   # 水平弧半长（米）= 0.01 ⇒ ±1cm
        start_feedback_iters: int = 2,           # 起始姿态的实测反馈轮数（0=关闭）
        start_settle_s: float = 1.0,             # 每段斜坡后的沉降等待
        enable_keyboard: bool = True,            # 录制脚本会传 False，自己管接管状态
        roi_spec=None,
        main_roi_spec=None,
        node_name: str = 'so101_rl_env',
        randomize_joint_half_deg: float = 1.5,
        full_frame_size=None,
        use_cameras=True,
    ):
        super().__init__()
        self.camera_keys = CAMERA_KEYS if use_cameras else ()
        self.full_frame_size = full_frame_size  # (width,height), opt-in recording only
        if full_frame_size and (roi_spec or main_roi_spec):
            raise ValueError("Full-frame capture cannot use ROI")
        self.image_shape = (full_frame_size[1], full_frame_size[0], 3) if full_frame_size else (image_size, image_size, 3)
        self.roi_spec = roi_spec
        self.main_roi_spec = main_roi_spec
        self.image_keys = list(self.camera_keys) + (['cam_main_roi'] if roi_spec and use_cameras else [])
        self.hz = float(hz)
        self.action_scale = float(action_scale)
        self.action_semantics = ACTION_SEMANTICS
        self._command_hold = CommandHold()
        self._last_joint_action = np.zeros(N_ACTION, dtype=np.float32)
        self._joint_control_status = {}
        # 起始位置：pan/lift/elbow 各自采样 ±1.5°，腕部补偿倾角。
        self.randomize_start = bool(randomize_start)
        self.randomize_half_m = float(randomize_half_m)  # legacy metadata only
        self.randomize_joint_half_deg = float(randomize_joint_half_deg)
        if not math.isfinite(self.randomize_joint_half_deg) or self.randomize_joint_half_deg < 0:
            raise ValueError("randomize_joint_half_deg must be finite and nonnegative")
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
        self._leader_stamp = 0.0
        self._camera_stats = {k: {'received': 0, 'accepted': 0, 'errors': 0,
                                 'arrival': 0.0, 'source_stamp_ns': 0,
                                 'last_error': '', 'valid': False,
                                 'first_arrival': 0.0, 'max_gap_s': 0.0,
                                 'pending_replaced': 0, 'duplicate_messages': 0,
                                 'processing_expired': 0, 'processing_delay_s': 0.0}
                              for k in self.camera_keys}
        self._last_obs_quality = {}
        self._image_workers = {
            cam: LatestFrameWorker(
                lambda msg, arrival, key=cam: self._process_image(msg, key, arrival),
                name=f'{cam}_latest_frame',
                on_error=lambda exc, key=cam: self._image_worker_error(key, exc))
            for cam in self.camera_keys}


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
        for cam in self.camera_keys:
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
                'command_error': gym.spaces.Box(-np.inf, np.inf, (N_ACTION,), dtype=f32),
                'hold_offset': gym.spaces.Box(-np.inf, np.inf, (N_ACTION,), dtype=f32),
            }),
            'images': gym.spaces.Dict({k: gym.spaces.Box(
                0, 255, self.image_shape, dtype=np.uint8)
                for k in self.image_keys}),
        })
        # 3D tool-origin action; four joints remain actuated.
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
            f'冻结 {list(FROZEN_JOINTS)}（录制/跟随阶段不下发；初始化另行归位）'
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

    def start_human_intervention(self):
        """Capture the differential anchors now, not on a later step."""
        now = time.monotonic()
        if (self.joint_pos is None or self.leader_pos is None or
                now-self._joint_stamp > .25 or now-self._leader_stamp > .25):
            raise RuntimeError('Fresh leader/follower feedback required to resume intervention')
        self._enter_mode('human', refresh=True)
        self.set_intervening(True)

    @property
    def is_intervening(self) -> bool:
        """人类是否正在接管。

        toggle 模式：按一下空格置 True，再按一下置 False。
        hold   模式：按住为 True，松开为 False。
        """
        return bool(self._intervening)

    def _enter_mode(self, mode: str, refresh=False):
        """切换控制模式并管理力矩（只在模式变化时执行一次）。

        ★ 进入人类接管时记下差分基准 (follower 已接受指令, leader 实测) —— 见 step()
          里模式 B 的说明。漏掉它会让从臂在接管瞬间跳回主臂所在姿态。
        """
        if mode == self._last_mode and not refresh:
            return
        if mode == 'human':
            # 人类接管：leader 必须松开，否则人手和电机较劲（抖振/发热）
            ok = self.set_leader_torque(False)
            if not ok:
                raise RuntimeError('Leader torque OFF failed; refusing intervention')
            # 从臂以已接受指令的几何目标为基准，不能混用尚有跟随滞后的
            # 实测角度，否则主臂不动也会在接管时产生回退指令。
            leader = self.leader_pos
            follower = self.joint_pos
            if leader is not None and follower is not None:
                self._lead_anchor = np.asarray(leader, float)[:N_ARM].copy()
                self._foll_anchor = (self._command_hold.target() - self._command_hold.offset)
                aligned=(self.leader_pose_in_follower_coordinates(leader)
                         if hasattr(self,'leader_pose_in_follower_coordinates') else np.asarray(leader))
                gap = float(np.abs(np.asarray(follower)[list(CONTROLLED_IDX)] - aligned[list(CONTROLLED_IDX)]).max())
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
            ok = self.set_leader_torque(True)
            if not ok:
                raise RuntimeError('Leader torque ON failed; refusing policy mode')
            self._lead_anchor = self._foll_anchor = None
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
            self._joint_stamp = time.monotonic()

    def _on_leader(self, msg: JointState):
        pos = self._reorder(msg)
        if pos.shape == (6,):
            with self._lock:
                self._leader_pos = pos
                self._leader_stamp = time.monotonic()

    def _on_image(self, msg: Image, key: str):
        arrival = time.monotonic()
        with self._lock:
            self._camera_stats[key]['received'] += 1
        if self._image_workers[key].submit(msg, arrival):
            with self._lock:
                self._camera_stats[key]['pending_replaced'] += 1

    def _image_worker_error(self, key, exc):
        with self._lock:
            stats = self._camera_stats[key]
            stats['errors'] += 1
            stats['valid'] = False
            stats['last_error'] = str(exc)
        self.node.get_logger().error(f'[{key}] image processing failed: {exc}')

    def _process_image(self, msg: Image, key: str, arrival):
        stamp = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        with self._lock:
            stats = self._camera_stats[key]
            previous_stamp = stats['source_stamp_ns']
            if stamp > 0 and stamp == previous_stamp:
                stats['duplicate_messages'] += 1
                return  # Reuse is normal; do not make an old frame look fresh.
        try:
            if stamp <= 0 or stamp < previous_stamp:
                raise ValueError('Missing or backwards camera timestamp')
            with self._lock:
                if time.monotonic() - arrival > 0.25:
                    stats['processing_expired'] += 1
                    return
            rgb = decode_rgb(msg)
            if self.full_frame_size and rgb.shape != self.image_shape:
                raise ValueError(f"Full-frame source shape {rgb.shape} != configured {self.image_shape}; no resizing allowed")
            images = {key: rgb.copy() if self.full_frame_size else cv2.resize(rgb, (self.img_size, self.img_size),
                                     interpolation=cv2.INTER_AREA)}
            if key == 'cam_main' and self.main_roi_spec:
                images['cam_main'] = crop_roi(rgb, self.main_roi_spec, self.img_size)
            if key == 'cam_main' and self.roi_spec:
                images['cam_main_roi'] = crop_roi(rgb, self.roi_spec, self.img_size)
        except Exception as exc:
            with self._lock:
                stats['errors'] += 1
                stats['valid'] = False
                stats['last_error'] = str(exc)
                errors = stats['errors']
            if errors == 1 or errors % 30 == 0:
                self.node.get_logger().error(f'[{key}] rejected frames={errors}: {exc}')
            return
        with self._lock:
            delay = time.monotonic() - arrival
            stats['processing_delay_s'] = delay
            if delay > 0.25:
                stats['processing_expired'] += 1
                return
            if stats['arrival']:
                stats['max_gap_s'] = max(stats['max_gap_s'], arrival - stats['arrival'])
            else:
                stats['first_arrival'] = arrival
            stats.update(arrival=arrival, source_stamp_ns=stamp, valid=True,
                         accepted=stats['accepted'] + 1, last_error='')
            self._images.update(images)  # full view and ROI become visible atomically

    def _quality_locked(self, max_age=0.25, max_skew=0.15):
        now = time.monotonic()
        cameras = {}
        problems = []
        for key, stats in self._camera_stats.items():
            age = now - stats['arrival'] if stats['arrival'] else None
            elapsed = now - stats['first_arrival'] if stats['first_arrival'] else 0
            cameras[key] = {**stats, 'age_s': age,
                            'fps': (stats['accepted'] - 1) / elapsed if elapsed > 0 else 0}
            if not stats['valid'] or age is None or age > max_age:
                problems.append(f'{key}: missing/invalid/stale ({age})')
        stamps = [v['source_stamp_ns'] for v in self._camera_stats.values()]
        skew = (max(stamps) - min(stamps)) / 1e9 if stamps else 0.0
        warnings = []
        if all(stamps) and skew > max_skew:
            warnings.append(f'camera skew {skew:.3f}s')
        for name, stamp in [('follower', self._joint_stamp), ('leader', self._leader_stamp)]:
            if not stamp or now - stamp > max_age:
                problems.append(f'{name}: stale joint feedback')
        return {'ok': not problems, 'problems': problems, 'cameras': cameras,
                'camera_skew_s': skew, 'max_camera_skew_s': max_skew,
                'warnings': warnings, 'camera_sync_required': False,
                'image_quality_policy': 'latest_per_camera_age_v2'}

    def observation_quality(self):
        with self._lock:
            return self._quality_locked()

    def last_observation_quality(self):
        with self._lock:
            return self._last_obs_quality

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
            self._last_obs_quality = self._quality_locked()
        if pos is None:
            pos = np.zeros(6, dtype=np.float32)
        if vel is None:
            vel = np.zeros(6, dtype=np.float32)
        images = {}
        for cam in self.image_keys:
            images[cam] = imgs.get(cam, np.zeros(self.image_shape, np.uint8))
        if self.display_image:
            try:
                show = np.concatenate([images[k] for k in self.image_keys], axis=1)
                cv2.imshow('so101_env', show[:, :, ::-1])   # 显示要 BGR
                cv2.waitKey(1)
            except Exception:
                pass
        idx=list(CONTROLLED_IDX)
        command_error=np.zeros(N_ACTION,np.float32)
        hold_offset=np.zeros(N_ACTION,np.float32)
        if self._command_hold.command is not None:
            command_error=(self._command_hold.target()[idx]-pos[idx]).astype(np.float32)
            hold_offset=self._command_hold.offset[idx].astype(np.float32)
        return {
            'state': {'joint_pos': pos.astype(np.float32), 'joint_vel': vel.astype(np.float32),
                      'command_error': command_error, 'hold_offset': hold_offset},
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
            self.node.get_logger().error(f'Torque service unavailable: {cli.srv_name}')
            return False
        req = SetBool.Request()
        req.data = bool(value)
        fut = cli.call_async(req)
        t0 = time.monotonic()
        while not fut.done() and time.monotonic() - t0 < timeout:
            time.sleep(0.02)
        if not fut.done():
            self.node.get_logger().error(f'Torque service timed out: {cli.srv_name}')
            return False
        result = fut.result()
        if result is None or not result.success:
            self.node.get_logger().error(
                f'Torque service rejected: {cli.srv_name}: {getattr(result, "message", "no response")}')
            return False
        return True

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
        q_to = np.asarray(q_to, dtype=float).reshape(-1)[:N_ARM].copy()
        if not np.isfinite(q_to).all():
            raise ValueError("Non-finite home target")
        q_to = np.clip(q_to, JOINT_LOW, JOINT_HIGH)
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
        # Do not cap duration: a cap would violate the requested peak speed.
        # Quintic profile peak acceleration = 5.774 * displacement / duration².
        dur = max(d_min, 1.875 * math.degrees(disp) / v_max_deg,
                  math.sqrt(5.774 * math.degrees(disp) / 60.0))

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

    def _settled_joint_position(self, motion_s=0.0):
        """Wait for fresh, stable feedback after the planned trajectory ends."""
        end = time.monotonic() + motion_s
        deadline = end + max(2.0, self.start_settle_s)
        samples = []
        last_stamp = 0.0
        while time.monotonic() < deadline:
            now = time.monotonic()
            with self._lock:
                stamp = self._joint_stamp
                pos = None if self._joint_pos is None else self._joint_pos[:N_ARM].copy()
            if pos is None or now - stamp > 0.25:
                raise RuntimeError('归位反馈过期，拒绝继续归位/录制')
            if now >= end and stamp != last_stamp:
                last_stamp = stamp
                samples.append((now, pos))
                samples = [(t, q) for t, q in samples if now - t <= 0.4]
                if len(samples) >= 4 and samples[-1][0] - samples[0][0] >= 0.3:
                    values = np.stack([q for _, q in samples])
                    if np.max(np.ptp(values, axis=0)) < math.radians(0.15):
                        return np.median(values, axis=0)
            time.sleep(0.02)
        raise RuntimeError('归位后反馈未稳定，拒绝开始录制')

    def _prepare_start_pose(self):
        """Backend-specific reset preparation; default backend preserves roll."""
        return None

    def reset(self, *, seed=None, options=None, wait_for_data: float = 5.0):
        """回到（随机化的）初始位姿并返回第一帧观测。

        起点 = START_JOINT_POSE，pan/lift/elbow 分别随机 ±1.5°，腕部补偿倾角。
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

        if self.joint_pos is None or time.monotonic() - self._joint_stamp > 0.25:
            raise RuntimeError('无新鲜关节反馈，拒绝归位')

        roll_home_info=self._prepare_start_pose()

        # 起点在名义关节姿态附近采样；不做 IK。
        q5 = np.asarray(START_JOINT_POSE, dtype=float)[:N_ARM].copy()
        # wrist_roll 使用实测值，归位不改变冻结维度。
        if self.joint_pos is not None:
            q5[4] = float(self.joint_pos[4])
        # Displacements describe the selected target relative to the nominal
        # start, not the measured reset residual. Fixed/fallback starts are zero.
        start_info = {'randomized': False, 'dx_m': 0.0, 'dy_m': 0.0, 'dz_m': 0.0,
                      'randomization_mode': 'fixed', 'joint_delta_deg': [0.0] * N_ARM,
                      'joint_half_deg': self.randomize_joint_half_deg}
        if self.randomize_start:
            q_rand, info = randomize_joint_start_pose(
                self._rng, half_deg=self.randomize_joint_half_deg,
                q_cur=self.joint_pos, kin=self._kin)
            if info.get('ok'):
                q5[:4] = np.asarray(q_rand)[:4]
                start_info = {'randomized': True, **{
                    k: info[k] for k in ('dx_m', 'dy_m', 'dz_m', 'err_mm',
                                         'tube_angle_deg', 'd_tilt_deg',
                                         'margin_rad', 'iters', 'randomization_mode',
                                         'joint_half_deg', 'joint_delta_deg')
                }}
            else:
                # 不下发越界或倾角不合格的随机目标，记录回退原因。
                self.node.get_logger().warn(
                    f'起始位置随机化未通过复核，本次退回名义起点：{info.get("reason")}')
                start_info['reason'] = info.get('reason')
        self.last_start_info = start_info
        start_info['wrist_roll_home'] = roll_home_info
        start_info['target_tilt_deg'] = tube_tilt_deg(q5)
        start_info['frozen_wrist_roll_deg'] = math.degrees(q5[4])
        if start_info['randomized']:
            self.node.get_logger().info(
                f'起始关节随机化 {start_info["joint_delta_deg"][:3]}°；dx {start_info["dx_m"]*100:+.2f}cm / '
                f'dy {start_info["dy_m"]*100:+.2f}cm / '
                f'dz {start_info["dz_m"]*100:+.2f}cm，'
                f'管子距竖直 {start_info["tube_angle_deg"]:.3f}°，'
                f'限位余量 {start_info["margin_rad"]:.2f}rad')

        # ★ 重力静差补偿：q5 是希望手臂**实际**停在的姿态，而舵机是比例环、
        #   发什么就停在什么下方几度。所以下发的**指令**要加上归位脚本收敛出来的
        #   那个偏置。冻结维度的偏置无所谓 —— _publish_arm_ramp 里会被实测值覆盖。
        q_base = np.clip(q5 + self.home_offset, JOINT_LOW, JOINT_HIGH)
        q_cmd = q_base.copy()

        # ★ 不发夹爪指令：夹爪是**冻结**的，一直夹着试管。
        #   踩坑（2026-09-29 排查发现）：原代码这里发 `_send_gripper(-1.0)`，
        #   而 GRIPPER_CLOSE_RAD = −0.2 rad，会把夹爪从保存的 +0.325 rad
        #   **压过去 0.5 rad** —— 每个 episode 都挤一次试管。既然夹爪不参与控制，
        #   就一个指令都不该发（它的位置由 ros2_control 的夹爪控制器保持）。
        #   将来若要放开夹爪，这里恢复 _send_gripper 即可。

        # 平滑走过去（不是弹过去）；冻结维度在 _publish_arm_ramp 里已锁在实测值
        dur = self._publish_arm_ramp(q_cmd, hold_idx=(4,))   # 只锁 wrist_roll
        q_measured = self._settled_joint_position(dur)

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
            q_act = q_measured
            e = q5 - q_act
            for i in (4, 5):                     # 冻结维度不修正
                if i < N_ARM:
                    e[i] = 0.0
            if float(np.abs(e).max()) < self.start_feedback_tol:
                break
            e = np.clip(e, -START_FEEDBACK_MAX, START_FEEDBACK_MAX)
            q_cmd = np.clip(q_cmd + 0.7 * e,
                            q_base - math.radians(6), q_base + math.radians(6))
            q_cmd = np.clip(q_cmd, JOINT_LOW, JOINT_HIGH)
            d2 = self._publish_arm_ramp(q_cmd, hold_idx=(4,))
            q_measured = self._settled_joint_position(d2)
            n_fb += 1
        start_info['feedback_iters'] = n_fb
        # ★ 报出"实际停在哪 vs 目标"的残差 —— phase1_home.py 一直有这个数字，
        #   reset 之前没有，于是"复位不如归位脚本准"只能靠感觉。现在能量了。
        cur = self.joint_pos
        if cur is not None:
            q_act = np.asarray(cur, dtype=float).reshape(-1)[:N_ARM]
            start_info['resid_deg'] = float(
                math.degrees(np.abs(q5 - q_act)[:4].max()))

        # ★ 复核：量**实际**停在哪、管子多歪。这是"重力补偿到底对不对"的唯一
        #   现场证据。补偿为 0（没跑归位脚本）时管子会歪到 ~8.8°，这里会告警。
        cur = self.joint_pos
        if cur is not None:
            q_act = np.asarray(cur, dtype=float).reshape(-1)[:N_ARM]
            start_info['act_tilt_deg'] = tube_tilt_deg(q_act)
            start_info['act_z_cm'] = tool_z(q_act) * 100.0
            if start_info['act_tilt_deg'] > 3.0:
                if start_info['target_tilt_deg'] > 3.0:
                    self.node.get_logger().warn(
                        f'当前冻结 wrist_roll={start_info["frozen_wrist_roll_deg"]:.2f}°，'
                        f'目标模型倾角本身为 {start_info["target_tilt_deg"]:.2f}°（超过 3°）；'
                        f'归位后模型倾角 {start_info["act_tilt_deg"]:.2f}°。'
                        '请检查冻结关节起始姿态，不能据此判定重力补偿失效。')
                else:
                    self.node.get_logger().warn(
                        f'起始模型倾角 {start_info["act_tilt_deg"]:.2f}° 超过 3°，'
                        f'目标倾角 {start_info["target_tilt_deg"]:.2f}°；'
                        '请检查实测关节残差、反馈和负载补偿。')
            else:
                self.node.get_logger().info(
                    f'起始姿态：管子距竖直 {start_info["act_tilt_deg"]:.2f}°，'
                    f'末端 z={start_info["act_z_cm"]:.1f}cm')
        residual = float(np.max(np.abs(q5 - q_measured)[:4]))
        start_info['resid_deg'] = math.degrees(residual)
        start_info['target_joint_rad'] = q5.tolist()
        start_info['measured_joint_rad'] = q_measured.tolist()
        start_info['actual_joint_delta_deg'] = np.degrees(
            q_measured - np.asarray(START_JOINT_POSE)[:N_ARM]).tolist()
        acceptance_tol = START_ACCEPTANCE_TOL
        settled_tilt = tube_tilt_deg(q_measured)
        start_info['precision_tol_deg'] = math.degrees(self.start_feedback_tol)
        start_info['acceptance_tol_deg'] = math.degrees(acceptance_tol)
        start_info['precision_met'] = residual <= self.start_feedback_tol
        start_info['act_tilt_deg'] = settled_tilt
        start_info['act_z_cm'] = tool_z(q_measured) * 100.0
        start_info['accepted'] = (residual <= acceptance_tol + 1e-12 and settled_tilt <= 3.0)
        start_info['accepted_outside_precision'] = (
            start_info['accepted'] and not start_info['precision_met'])
        if not start_info['accepted']:
            raise RuntimeError(f'归位未通过验收: residual={math.degrees(residual):.2f}°, '
                               f'limit={math.degrees(acceptance_tol):.3f}°, '
                               f'tilt={settled_tilt:.2f}°（上限 3.00°）；'
                               f'target_tilt={start_info["target_tilt_deg"]:.2f}°, '
                               f'frozen_roll={start_info["frozen_wrist_roll_deg"]:.2f}°；'
                               '请检查目标姿态、冻结关节、归位反馈与补偿')
        if start_info['accepted_outside_precision']:
            self.node.get_logger().warn(
                f'归位稳定，在放宽后的验收范围内：'
                f'残差 {math.degrees(residual):.3f}°，'
                f'修正目标 {math.degrees(self.start_feedback_tol):.3f}°，'
                f'验收上限 {math.degrees(acceptance_tol):.3f}°；已记录起点质量')
        self._command_hold.calibrate(q_cmd, q_measured)
        start_info['hold_command_rad'] = q_cmd.tolist()
        start_info['hold_offset_rad'] = self._command_hold.offset.tolist()
        return self._get_obs(), {'succeed': False, 'start': start_info}

    def set_intervene_deadband(self, rad: float):
        """保留旧录制器接口；关节干预记录请求的归一化增量，不做死区过滤。"""
        self.intervene_deadband = float(rad)
        return self.intervene_deadband

    def get_intervene_action(self, deadband=None):
        """Expert and policy both store normalized requested relative joint increments."""
        return self._last_joint_action.copy() if self.is_intervening else None

    def begin_control_cycle(self):
        """Include caller observation/inference time in the next step's period."""
        self._caller_cycle_started = time.monotonic()

    def step(self, action: np.ndarray):
        step_started = time.monotonic()
        cycle_started = getattr(self, '_caller_cycle_started', None)
        self._caller_cycle_started = None
        if cycle_started is None:
            cycle_started = step_started
        delta = bounded_joint_delta(action, self.action_scale)
        cur = self.joint_pos
        if cur is None or not np.all(np.isfinite(cur)) or time.monotonic() - self._joint_stamp > .25:
            raise RuntimeError('No fresh joint feedback; refusing step')
        target5 = self._command_hold.target()
        self._enter_mode('human' if self.is_intervening else 'policy')
        idx=list(CONTROLLED_IDX)
        source_problem=None
        human_goal=None
        reference=target5-self._command_hold.offset
        if self.is_intervening:
            leader=self.leader_pos
            if leader is None or not np.all(np.isfinite(leader)) or time.monotonic()-self._leader_stamp>.25:
                source_problem='stale_leader';delta=np.zeros(N_ACTION)
            elif self._lead_anchor is None or self._foll_anchor is None:
                source_problem='missing_human_anchor';delta=np.zeros(N_ACTION)
            else:
                desired=self._foll_anchor+(np.asarray(leader)[:N_ARM]-self._lead_anchor)
                human_goal=desired[idx].tolist()
                delta=desired[idx]-reference[idx]  # Shared executor clips both expert and policy requests.
        request=delta/self.action_scale if self.is_intervening else np.asarray(action,float)
        low=getattr(self,'_action_low',JOINT_LOW[idx])
        high=getattr(self,'_action_high',JOINT_HIGH[idx])
        candidate,status=execute_joint_request(target5[idx],cur[idx],self._command_hold.offset[idx],
            request,low,high,self.action_scale)
        target5[idx]=candidate
        self._last_joint_action=np.asarray(status['requested_action'],np.float32)
        status['source']='human' if self.is_intervening else 'policy'
        status['step_started_monotonic_s']=step_started
        status['leader_feedback_monotonic_s']=self._leader_stamp
        status['follower_feedback_monotonic_s']=self._joint_stamp
        if human_goal is not None:status['human_goal_geometric_rad']=human_goal
        if source_problem:status.update(accepted=False,reason=source_problem)
        self._joint_control_status=status
        if source_problem or status['limited']:
            now=time.monotonic()
            if now-getattr(self,'_last_joint_warning',0.)>=1.:
                self.node.get_logger().warn(f'关节控制受限：{source_problem or status["joint_reasons"]}')
                self._last_joint_warning=now
        target5 = hold_frozen(target5, cur)
        self._publish_arm(target5)
        status['published_monotonic_s']=time.monotonic()
        self._command_hold.remember(target5)
        if not self.is_intervening:
            self._drive_leader(cur)

        # 维持控制周期
        time.sleep(max(0., 1.0 / self.hz - (time.monotonic() - cycle_started)))

        obs = self._get_obs()
        reward = self.compute_reward(obs)
        self._step_count += 1
        self._episode_return += reward
        done = self._step_count >= self.max_episode_steps or reward == 1

        info = {'succeed': bool(reward), 'intervening': self.is_intervening,
                'observation_quality': self.last_observation_quality(),
                'joint_control': self._joint_control_status}
        status['step_finished_monotonic_s']=time.monotonic()
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
        for worker in getattr(self, '_image_workers', {}).values():
            worker.close()
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
