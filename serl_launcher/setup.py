from setuptools import setup, find_packages

# ── 与原版的差异（重要，修掉了两个会让环境崩掉的坑）────────────────────
#
# 1. 删掉 "typing" —— 这是阶段 A 那次 "pip/conda/ros2 全部崩溃" 的根因！
#    PyPI 上的 `typing` 是个 Python 2/3.7 时代的 backport（3.7.4.3）。
#    它会在 site-packages 写出 typing.py，遮蔽 stdlib 的 typing。
#    在 Python 3.10 下必崩：
#        AttributeError: type object 'Callable' has no attribute '_abc_registry'
#    typing 从 Python 3.5 起就是标准库的一部分，**不应该出现在依赖里**。
#
# 2. "zmq" → "pyzmq" —— PyPI 上的 `zmq` 是个 0.0.0 的空壳占位包，
#    真正提供 zmq 模块的是 pyzmq。agentlace 用的也是 pyzmq。
#
# 3. agentlace 仓库已从 youliangtan/agentlace 迁移到 assistive-autonomy/agentlace。
#    旧 commit cf2c337 (2024-06-17) 已实测仍可 clone，保留原 pin 以保证兼容；
#    若将来失效，改用：agentlace@git+https://github.com/assistive-autonomy/agentlace.git

setup(
    name="serl_launcher",
    version="0.1.2",
    description="library for rl experiments",
    url="https://github.com/rail-berkeley/serl",
    author="auth",
    license="MIT",
    install_requires=[
        "pyzmq",              # 原为 "zmq"（空壳包）
        # "typing",           # ← 已删除：Python 2 时代 backport，会破坏 stdlib
        "typing_extensions",
        "opencv-python",
        "lz4",
        "cloudpickle",
        "agentlace@git+https://github.com/youliangtan/agentlace.git@cf2c337c5e3694cdbfc14831b239bd657bc4894d",
    ],
    packages=find_packages(),
    zip_safe=False,
)
