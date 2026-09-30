#!/usr/bin/env python3
"""Non-blocking keyboard input for the SO101 scripts.

Why this exists (and why not pynput / /dev/input)
------------------------------------------------
On this machine the only keyboard path that works from a script is
termios cbreak + select(2) on stdin. Everything else was tried and failed
(see 交接文档 §2.4):
  * pynput xorg backend  -> needs an X server; DISPLAY=127.0.0.1:0.0 does not
                            connect (the X server is only on the unix socket, :0)
  * pynput uinput       -> that version is missing pynput.mouse._uinput
  * /dev/input          -> does not exist in this environment

So: one shared implementation, used by both `phase1_home.py` (abort key during
homing) and `record_demo_so101.py` (phase keys during recording). It used to be
duplicated inside phase1_home.py; keeping two copies is how "changed one, forgot
the other" bugs happen.

WARNING
-------
termios changes the terminal mode. If the process is killed with `kill -9` the
terminal can be left in cbreak; run `stty sane` to recover. A normal exit (or a
KeyboardInterrupt handled by the caller) restores it via __exit__.
"""

import select
import sys
import termios
import time
import tty

# 放回字符的存活时间（秒）。超过就丢掉 —— 见 read_key() 的说明。
PUSHBACK_TTL = 1.0

# Keys that the legacy `pressed()` helper treats as "a key was pressed".
LEGACY_KEYS = (' ', 'q', 'Q', '\x1b')


class NonBlockingKey:
    """Read single keypresses from a TTY without blocking.

    Usage:
        with NonBlockingKey() as kb:
            if not kb.ok:
                # stdin is not a TTY -- refuse to run a human-in-the-loop script
                ...
            k = kb.read_key()          # -> single char, or None if nothing
    """

    def __init__(self):
        self.ok = False
        self.fd = None
        self.old = None
        # unread() 放回来的键：(字符, 放回时间)。**必须带时间戳**，
        # 否则会永久遮蔽后续输入 —— 见 read_key() 里那段踩坑记录。
        self._pushback = []
        if not sys.stdin.isatty():
            return
        try:
            self.fd = sys.stdin.fileno()
            self.old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
            self.ok = True
        except Exception:
            self.ok = False

    def read_key(self):
        """Return one character, or None if no key is waiting.

        Once consumed the character is gone; do not mix read_key() and pressed()
        in the same process (pressed() also consumes).
        """
        if not self.ok:
            return None
        # ⚠️⚠️ 踩坑（2026-09-29，真机把录制卡死，只能 Ctrl-C 丢数据）：
        #   最初写成"如果有放回字符就直接返回它"，结果 ——
        #     某个阶段不接受这个字符 → unread() 放回 → 下一次又立刻返回它 →
        #     再放回 …… **select() 从此再也不会被调用**，
        #   进程对键盘**永久失聪**（实测：终端输入队列里积了 9 个未读字节，
        #   而进程还在正常循环 —— 就是没在读）。
        #   一个误按的回车就能触发，而且外部无法解救（TIOCSTI 注入也读不到）。
        # 对策两条，缺一不可：
        #   ① 每次都先问内核"有新键吗"；有新键就**丢弃过期的放回字符**、读新键
        #   ② 放回字符只在 PUSHBACK_TTL 秒内有效，过期就丢
        ready = False
        try:
            ready = bool(select.select([self.fd], [], [], 0)[0])
        except Exception:
            ready = False
        if ready:
            self._pushback.clear()          # ① 有新键 → 旧放回字符作废
            try:
                return sys.stdin.read(1)
            except Exception:
                return None
        if self._pushback:
            ch, t = self._pushback[-1]
            if time.time() - t < PUSHBACK_TTL:   # ② 只在还新鲜时返回
                return ch
            self._pushback.clear()
        return None

    def unread(self, ch):
        """Put a character back so a later read_key() returns it again.

        Why: a phase that only cares about some keys (e.g. the automatic leader
        alignment, which only watches for `q`) must not SWALLOW the others --
        otherwise a `c` pressed a moment too early is silently lost and the user
        has to press it again for no visible reason.
        """
        if ch:
            self._pushback.append((ch, time.time()))

    def reassert(self):
        """Re-apply cbreak mode.

        Needed after SIGTSTP (Ctrl-Z) + `fg`: bash restores its OWN terminal modes
        when the job is stopped, so the process comes back in **canonical** mode and
        single keypresses are no longer delivered until a newline is typed
        (实测：fg 之后 isig/icanon/echo 全被还原). Call this at every phase change.
        """
        if not self.ok:
            return
        try:
            tty.setcbreak(self.fd)
        except Exception:
            pass

    def pressed(self) -> bool:
        """True if a "legacy" key (space / q / Q / ESC) was pressed.

        Kept for backward compatibility with phase1_home.py, which uses it as a
        plain abort signal. It CONSUMES the character.
        """
        return self.read_key() in LEGACY_KEYS

    def drain(self):
        """Discard any pending keypresses (call before a new phase)."""
        while self.read_key() is not None:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        if self.ok and self.old is not None:
            try:
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)
            except Exception:
                pass
        return False
