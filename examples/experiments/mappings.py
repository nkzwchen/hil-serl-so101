"""实验配置映射表。

原版写法是直接 import 全部实验配置，但这些配置大多依赖 franka_env（需要 ROS1 rospy），
在本机（ROS2 + SO101）环境下 import 会抛 ModuleNotFoundError，
导致**整个模块导入失败**，进而 train_rlpd.py 连启动都不行。

这里改成「宽容加载」：
  · 能用就注册；
  · 依赖缺失（如 franka_env）就跳过并记录原因，不影响其他实验可用。
用法不变：CONFIG_MAPPING["dummy_test"] / ["ram_insertion"] ...
"""

import importlib
import sys

# 实验名 -> "模块路径:TrainConfig 类名"
_EXPERIMENT_MODULES = {
    # 本项目（SO101，不依赖 ROS1）
    "dummy_test": "experiments.dummy_test.config:TrainConfig",
    # 真实 SO101（需要 ROS2 链路在跑；env 是 ROS2 薄客户端）
    "so101": "experiments.so101.config:TrainConfig",
    # 以下来自 HIL-SERL 原始示例，依赖 franka_env（ROS1）。
    # 在 SO101 环境下会加载失败——这是预期行为，会被静默跳过。
    "ram_insertion": "experiments.ram_insertion.config:TrainConfig",
    "usb_pickup_insertion": "experiments.usb_pickup_insertion.config:TrainConfig",
    "object_handover": "experiments.object_handover.config:TrainConfig",
    "egg_flip": "experiments.egg_flip.config:TrainConfig",
}

CONFIG_MAPPING = {}
LOAD_ERRORS = {}

for _name, _target in _EXPERIMENT_MODULES.items():
    _module_path, _class_name = _target.split(":")
    try:
        _mod = importlib.import_module(_module_path)
        CONFIG_MAPPING[_name] = getattr(_mod, _class_name)
    except Exception as _e:  # noqa: BLE001 - 故意吞掉，记录后继续
        LOAD_ERRORS[_name] = f"{type(_e).__name__}: {_e}"

if LOAD_ERRORS:
    # franka_env 相关失败是预期的（ROS1 依赖），不必打扰用户；
    # 其他失败则展开，因为那可能是真 bug。
    _unexpected = {n: m for n, m in LOAD_ERRORS.items() if "franka_env" not in m}
    if _unexpected:
        print("[mappings] 以下实验配置加载失败（非预期）：", file=sys.stderr)
        for _n, _msg in _unexpected.items():
            print(f"[mappings]   {_n}: {_msg}", file=sys.stderr)
    else:
        print(
            f"[mappings] 已跳过 {len(LOAD_ERRORS)} 个依赖 franka_env(ROS1) 的实验配置"
            f"（SO101 环境下正常）：{', '.join(LOAD_ERRORS)}",
            file=sys.stderr,
        )
