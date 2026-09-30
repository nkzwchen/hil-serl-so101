#!/usr/bin/env python3
"""诊断：从臂力矩到底有没有真的关掉。

背景：`/follower/set_torque {data: false}` 返回 success=True，但机械臂仍然很紧。
      怀疑 `_disable_torque` 的写入被静默忽略，需要读伺服寄存器的**真实值**来判定。

关键线索（lerobot feetech 控制表）：
    Torque_Enable  addr=40 (0x28)
    Lock           addr=55 (0x37)
    Goal_Position  addr=42 (0x2A)
  `FeetechMotorsBus._disable_torque` 的顺序是：
      1. 写 Torque_Enable = 0
      2. 写 Lock = 0
  如果 Torque_Enable 落在 Lock 保护的 EEPROM 区间内，第 1 步会被**静默忽略**，
  于是力矩根本关不掉 —— 这与观察到的现象一致。

⚠️ 本脚本会独占串口，运行前**必须先停掉 launch**（终端 A 按 Ctrl+C），
   否则串口被桥占用会报错。

用法：
    python free_arm.py            # 读力矩状态 + 关力矩 + 复验 + 打印关节角
    python free_arm.py --read     # 只读，不改任何东西
"""

import sys
from pathlib import Path

from lerobot.robots.so_follower import SOFollower, SOFollowerConfig

PORT = '/dev/ttyACM1'
ID = 'so101_follower_main'
CALIB_DIR = Path('/home/nkzwchen/.cache/huggingface/calibration/robots/so_follower')


def show_torque(robot, tag):
    print(f'--- Torque_Enable 实际值（{tag}）---')
    vals = {}
    for name in robot.bus.motors:
        try:
            v = robot.bus.read('Torque_Enable', name)
            vals[name] = v
            flag = 'ON (紧)' if v else 'OFF (松)'
            print(f'  {name:14s} = {v}   {flag}')
        except Exception as e:
            print(f'  {name:14s} 读取失败: {type(e).__name__}: {e}')
    return vals


def main():
    read_only = '--read' in sys.argv

    cfg = SOFollowerConfig(
        port=PORT, id=ID, calibration_dir=CALIB_DIR,
        use_degrees=True, disable_torque_on_disconnect=True,
    )
    robot = SOFollower(cfg)
    print(f'连接 {PORT} (id={ID}) ...')
    robot.connect()
    print('已连接\n')

    before = show_torque(robot, '操作前')

    if not read_only:
        print('\n>>> 调用 disable_torque()')
        robot.bus.disable_torque()
        after = show_torque(robot, '操作后')

        changed = {k: (before.get(k), after.get(k)) for k in before
                   if before.get(k) != after.get(k)}
        print()
        if all(v == 0 for v in after.values() if v is not None):
            print('✅ 所有电机 Torque_Enable 已为 0 —— 力矩确实关掉了')
            print('   （若仍感觉紧，那是 STS3215 齿轮箱的正常阻力，需要稍微用力；')
            print('     也请确认机械臂没有卡在某个自锁位形）')
        elif changed:
            print('⚠️ 部分电机状态变了但仍非 0：')
            for k, (a, b) in changed.items():
                print(f'   {k}: {a} -> {b}')
        else:
            print('❌ 所有电机状态**完全没变** —— disable_torque 的写入被忽略了！')
            print('   很可能是 Lock(addr=55) 保护了 Torque_Enable(addr=40)。')
            print('   下一步：先写 Lock=0，再写 Torque_Enable=0，然后复验。')

    print('\n--- 当前 Present_Position（原始 ticks）---')
    try:
        pos = robot.bus.sync_read('Present_Position')
        for k, v in pos.items():
            print(f'  {k:14s} = {v}')
    except Exception as e:
        print(f'  读取失败: {type(e).__name__}: {e}')

    print('\n--- get_observation() ---')
    try:
        for k, v in robot.get_observation().items():
            print(f'  {k} = {v}')
    except Exception as e:
        print(f'  失败: {type(e).__name__}: {e}')

    robot.disconnect()
    print('\n已断开。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
