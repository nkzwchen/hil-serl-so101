"""Separate-process Tk control window. GUI never touches robot or network objects."""
import multiprocessing as mp
import time

PHASES = {
 'CHECK': ('前置检查', '', '正在检查设备，请稍候'),
 'PRECISE': ('归位 / 对齐', 'x', '机械臂可能自动运动；退出需等待当前操作返回'),
 'READY': ('准备就绪', 'cqx', 'C 开始本段 · Q 跳过当前起点 · X 退出'),
 'POLICY': ('检查模型版本', '', '段间检查新权重，请稍候'),
 'RECORD': ('录制中', ' dqxh', '空格 切换接管 · D 结束本段 · Q 排除本段 · X 退出'),
 'RECORD_MANUAL': ('人工录制中', 'dqxh', 'D 结束本段 · Q 排除本段 · X 退出；主臂控制从臂'),
 'RETURN': ('人工回程', 'cqx', '主臂仍控制从臂；回到起点附近后按 C · Q 切换排除'),
 'SAVE': ('保存数据', '', '正在保存本地原始数据'),
 'REVIEW': ('标注 / 上传', '', '请操作标注窗口；确认上传或跳过本段上传'),
}


class KeyGate:
    def __init__(self):
        self.generation=0; self.phase='CHECK'; self.pending=[]; self.closed=False; self.events=[]
    def set_phase(self, phase):
        self.generation+=1; self.phase=phase; self.pending.clear()
    def receive(self, event):
        if event[0]=='close':self.closed=True
        elif event[0]=='key':
            _, generation, stamp, key=event
            if generation==self.generation and 0<=time.monotonic()-stamp and key and key in PHASES[self.phase][1]:
                self.pending.append((key,stamp))
    def read(self):
        if self.closed:return 'x'
        while self.pending:
            key,stamp=self.pending.pop(0)
            now=time.monotonic()
            self.events.append(dict(key=key, phase=self.phase, generation=self.generation,
                                    pressed_monotonic_s=stamp, handled_monotonic_s=now,
                                    dispatch_latency_ms=(now-stamp)*1000))
            return key
        return None


def _text_image(root, text, size=18, color='#222', width=640):
    """Render with FreeType: some conda Tk builds expose only X11 'fixed'."""
    from pathlib import Path
    from PIL import Image, ImageDraw, ImageFont, ImageTk
    paths = (
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/SimHei.ttf',
    )
    path = next((p for p in paths if Path(p).is_file()), None)
    if path is None:
        raise RuntimeError('Chinese font missing; install fonts-wqy-zenhei')
    font = ImageFont.truetype(path, size)
    lines = []
    for paragraph in text.split('\n'):
        line = ''
        for char in paragraph:
            if line and font.getlength(line + char) > width:
                lines.append(line); line = ''
            line += char
        lines.append(line)
    line_height = size + 8
    w = max(1, min(width, int(max(font.getlength(line) for line in lines)) + 4))
    rgb = tuple(v // 257 for v in root.winfo_rgb(root.cget('background')))
    im = Image.new('RGB', (w, line_height * len(lines) + 4), rgb)
    draw = ImageDraw.Draw(im)
    for i, line in enumerate(lines):
        draw.text(((w-font.getlength(line))/2, i*line_height), line, font=font, fill=color)
    return ImageTk.PhotoImage(im, master=root)


def event_key(event):
    # KeyRelease.char can be empty on X11; match press/release by keysym.
    symbol = event.keysym.lower()
    return ' ' if symbol == 'space' else (symbol if symbol in ('c','d','q','x','h') else '')


def _window(conn):
    try:
        import tkinter as tk
        root=tk.Tk();root.title('SO101 录制控制');root.geometry('680x430')
        def set_text(widget, text, size=18, color='#222'):
            widget._text_image = _text_image(root, text, size, color)
            widget.configure(image=widget._text_image)
        focus_status=tk.Label(root,name='keyboard_focus');focus_status.pack(pady=(12,0))
        phase=tk.Label(root);phase.pack(pady=18)
        mode=tk.Label(root);mode.pack()
        detail=tk.Label(root);detail.pack(pady=14)
        set_text(phase, '正在连接录制器', 28)
        help_label=tk.Label(root);help_label.pack()
        set_text(help_label, '点击本窗口后按键；按钮也可操作。\n窗口操作不是硬件急停，阻塞中的设备操作需先返回。', 16, '#666')
        feedback=tk.Label(root);feedback.pack()
        bar=tk.Frame(root);bar.pack(pady=10)
        state={'phase':'CHECK','generation':0,'closing':False}
        held=set(); releases={};buttons={}
        def send(key):
            if not state['closing'] and key in PHASES[state['phase']][1]:
                stamp=time.monotonic()
                conn.send(('key',state['generation'],stamp,key))
                set_text(feedback, f'{"空格" if key==" " else key.upper()} 已收到，正在处理', 16, '#167a55')
        for key,label in [('c','C 继续'),(' ','空格 接管'),('d','D 结束'),('q','Q 排除/跳过'),('x','X 退出')]:
            b=tk.Button(bar,command=lambda k=key:send(k),takefocus=False)
            set_text(b,label,17)
            b.pack(side='left',padx=3);buttons[key]=b
        def press(event):
            key=event_key(event)
            if not key or key not in ' cdqxh':return
            if key in releases:root.after_cancel(releases.pop(key))
            if key not in held:held.add(key);send(key)
            return 'break'
        def release(event):
            key=event_key(event)
            if not key:return
            if key in releases:root.after_cancel(releases[key])
            releases[key]=root.after(60,lambda: (held.discard(key),releases.pop(key,None)))
        root.bind_class('ControlKeys','<KeyPress>',press)
        root.bind_class('ControlKeys','<KeyRelease>',release)
        # Route before Tk Button class bindings (space must never click a focused C button).
        for widget in [root,*buttons.values()]:
            widget.bindtags(('ControlKeys',)+widget.bindtags())
        def clear_held(event=None):
            for timer in releases.values():root.after_cancel(timer)
            releases.clear();held.clear()
        root.bind('<FocusOut>',clear_held,add='+')
        # Clicking text/background must also restore keyboard focus to this window.
        root.bind('<Button-1>',lambda event:root.focus_set(),add='+')
        def close():
            state['closing']=True;conn.send(('close',))
            set_text(detail,'已请求退出，等待当前操作返回；不会通过关窗强制终止设备调用。')
            for b in buttons.values():b.configure(state='disabled')
        root.protocol('WM_DELETE_WINDOW',close)
        last_focus=[None]
        def refresh_focus():
            focused=root.focus_displayof()
            active=focused is not None and focused.winfo_toplevel()==root
            if active != last_focus[0]:
                last_focus[0]=active
                text=('键盘焦点：在本窗口 · 可按当前阶段快捷键' if active
                      else '键盘焦点：不在本窗口 · 请先点击这里')
                focus_status.focus_active=active
                set_text(focus_status,text,20,'#167a55' if active else '#b45309')
        def tick():
            try:
                refresh_focus()
                while conn.poll():
                    msg=conn.recv()
                    if msg[0]=='stop':root.destroy();return
                    if msg[0]=='ack':
                        _,generation,key,latency=msg
                        if generation==state['generation']:
                            set_text(feedback,f'{"空格" if key==" " else key.upper()} 主程序已处理 · {latency:.0f} ms',16,'#167a55')
                        continue
                    _,p,g,extra=msg
                    set_text(feedback,'',16)
                    state.update(phase=p,generation=g)
                    set_text(phase,PHASES[p][0],28);set_text(mode,extra,21,'#167a55')
                    if p=='READY':root.lift();root.focus_force()
                    if not state['closing']:set_text(detail,PHASES[p][2])
                    for key,b in buttons.items():b.configure(state='normal' if key in PHASES[p][1] and not state['closing'] else 'disabled')
                root.after(40,tick)
            except (EOFError,BrokenPipeError):root.destroy()
        conn.send(('ready',));root.after(40,tick);root.mainloop()
    except Exception as exc:
        try:conn.send(('error',str(exc)))
        except (EOFError,BrokenPipeError):pass
    finally:conn.close()


class ControlPanel:
    def __init__(self):
        self.ok=False;self.gate=KeyGate()
        ctx=mp.get_context('spawn');self.conn,child=ctx.Pipe()
        self.process=ctx.Process(target=_window,args=(child,),daemon=True);self.process.start();child.close()
        if not self.conn.poll(10):
            self.__exit__();raise RuntimeError('控制窗口启动超时；可使用 --terminal-controls')
        msg=self.conn.recv()
        if msg[0]!='ready':
            self.__exit__();raise RuntimeError(f'控制窗口无法启动：{msg}；可使用 --terminal-controls')
        self.ok=True
    def _receive(self):
        if not self.process.is_alive():raise KeyboardInterrupt('Control window exited')
        while self.conn.poll():
            msg=self.conn.recv()
            if msg[0]=='error':raise RuntimeError(msg[1])
            self.gate.receive(msg)
    def check(self):
        self._receive()
        if self.gate.closed:raise KeyboardInterrupt('Control window requested exit')
    def set_phase(self,phase,detail=''):
        self._receive();self.gate.set_phase(phase)
        if self.gate.events:
            event=self.gate.events[-1]
            if 'next_phase' not in event:
                event.update(next_phase=phase, phase_latency_ms=(time.monotonic()-event['pressed_monotonic_s'])*1000)
                print(f'[按键阶段] {event["key"]!r} → {phase}: {event["phase_latency_ms"]:.1f} ms',flush=True)
        self.conn.send(('state',phase,self.gate.generation,detail))
    def read_key(self):
        self._receive();key=self.gate.read()
        if key is not None and not self.gate.closed:
            event=self.gate.events[-1]
            self.conn.send(('ack',self.gate.generation,key,event['dispatch_latency_ms']))
            print(f'[按键处理] {key!r} phase={event["phase"]}: {event["dispatch_latency_ms"]:.1f} ms',flush=True)
        return key
    def mark_control_step(self, intervening):
        # Completion of first step after a toggle, including the torque RPC.
        for event in reversed(self.gate.events):
            if event['key']==' ' and 'control_step_latency_ms' not in event:
                event.update(control_step_latency_ms=(time.monotonic()-event['pressed_monotonic_s'])*1000,
                             intervening_after_step=intervening)
                print(f'[接管执行] human={intervening}: {event["control_step_latency_ms"]:.1f} ms（首步完成）',flush=True)
                break
    def timing_events(self):
        return [dict(event) for event in self.gate.events]
    def unread(self,key):pass  # Wrong-phase input must never trigger the next phase.
    def reassert(self):pass
    def drain(self):self._receive();self.gate.pending.clear()
    def __enter__(self):return self
    def __exit__(self,*args):
        try:self.conn.send(('stop',))
        except (EOFError,BrokenPipeError,OSError):pass
        self.process.join(timeout=2)
        if self.process.is_alive():self.process.terminate();self.process.join(timeout=2)
        self.conn.close();self.ok=False
        return False


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--demo',action='store_true',required=True)
    ap.parse_args()
    with ControlPanel() as panel:
        panel.set_phase('READY','界面演示 · 不连接设备')
        while True:
            k=panel.read_key()
            if k=='x':break
            if k=='c':panel.set_phase('RECORD','演示：模型控制')
            elif k==' ':panel.set_phase('RECORD','演示：人工接管')
            elif k in ('d','q'):panel.set_phase('RETURN','演示：人工回程')
            time.sleep(.03)
