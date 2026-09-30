#!/usr/bin/env python
"""直接测试 SO101Leader 的总线写入能力（不经 ROS2）。

目的：定位 `KeyError: 'shoulder_pan.pos'` 到底出在哪。
会尝试多种调用方式，看哪种能成功。
"""
import traceback
from pathlib import Path

from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

cfg = SO101LeaderConfig(
    port='/dev/ttyACM0',
    id='so101_leader_main',
    calibration_dir=Path('/home/nkzwchen/.cache/huggingface/calibration/teleoperators/so_leader'),
    use_degrees=True,
)
r = SO101Leader(cfg)
print('连接 leader /dev/ttyACM0 ...')
r.connect(calibrate=False)
print('  ✅ 已连接')

print('\n--- 1) get_action() 的键名 ---')
act = r.get_action()
print('  ', act)

print('\n--- 2) bus.motors 的键名 ---')
print('  ', list(r.bus.motors.keys()))

print('\n--- 3) 尝试 sync_write("Goal_Position", {"shoulder_pan": v}) ---')
try:
    r.bus.sync_write('Goal_Position', {'shoulder_pan': act['shoulder_pan.pos']})
    print('  ✅ 成功（用 motor 名，无 .pos 后缀）')
except Exception as e:
    print(f'  ❌ {type(e).__name__}: {e}')
    traceback.print_exc()

print('\n--- 4) 尝试 sync_write("Goal_Position", {"shoulder_pan.pos": v}) ---')
try:
    r.bus.sync_write('Goal_Position', {'shoulder_pan.pos': act['shoulder_pan.pos']})
    print('  ✅ 成功（带 .pos 后缀）')
except Exception as e:
    print(f'  ❌ {type(e).__name__}: {e}')

print('\n--- 5) 尝试 sync_write 全 6 关节（motor 名）---')
try:
    vals = {k.removesuffix('.pos'): v for k, v in act.items()}
    r.bus.sync_write('Goal_Position', vals)
    print('  ✅ 成功:', vals)
except Exception as e:
    print(f'  ❌ {type(e).__name__}: {e}')

print('\n--- 6) normalized_data 包含哪些寄存器 ---')
print('  ', getattr(r.bus, 'normalized_data', '<无此属性>'))

r.disconnect()
print('\n已断开')
