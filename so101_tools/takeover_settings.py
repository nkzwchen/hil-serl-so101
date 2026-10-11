"""Persistent UI preference and per-episode takeover accounting."""
import json
import os
from pathlib import Path

SETTINGS_PATH = Path.home()/'.config/so101/control_panel.json'

def validate_limit(value):
    if type(value) is not int or not 0 <= value <= 999:
        raise ValueError('接管上限请输入 0～999 的整数；0 表示不限')
    return value

def load_limit(path=None):
    path=Path(path) if path is not None else SETTINGS_PATH
    try:return validate_limit(json.loads(path.read_text())['max_takeovers'])
    except (OSError,ValueError,KeyError,TypeError):return 0

def save_limit(value,path=None):
    value=validate_limit(value);path=Path(path) if path is not None else SETTINGS_PATH
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    try:
        tmp.write_text(json.dumps({'max_takeovers':value},ensure_ascii=False)+'\n')
        tmp.replace(path)
    finally:
        if tmp.exists():tmp.unlink()

class TakeoverBudget:
    def __init__(self,limit=0):
        self.limit=validate_limit(limit);self.count=0;self.limit_reached=False
    def press_space(self,currently_human):
        if currently_human:return False  # Hand back to policy; no new takeover.
        if self.limit and self.count>=self.limit:
            self.limit_reached=True
            return True  # Equivalent to D; no additional RECORD action.
        self.count+=1
        return False
    def detail(self,human):
        return f'{"人工接管" if human else "模型控制"} · 接管 {self.count}/{self.limit or "不限"}'
