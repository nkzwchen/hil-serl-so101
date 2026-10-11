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
        self.generation=0; self.phase='CHECK'; self.pending=[]; self.closed=False
    def set_phase(self, phase):
        self.generation+=1; self.phase=phase; self.pending.clear()
    def receive(self, event):
        if event[0]=='close':self.closed=True
        elif event[0]=='key':
            _, generation, stamp, key=event
            if generation==self.generation and 0<=time.monotonic()-stamp<1 and key and key in PHASES[self.phase][1]:
                self.pending.append((key,stamp))
    def read(self):
        if self.closed:return 'x'
        while self.pending:
            key,stamp=self.pending.pop(0)
            if time.monotonic()-stamp<1:return key
        return None


def _window(conn):
    try:
        import tkinter as tk
        root=tk.Tk();root.title('SO101 录制控制');root.geometry('680x370')
        phase=tk.StringVar(value='正在连接录制器'); detail=tk.StringVar(); mode=tk.StringVar()
        tk.Label(root,textvariable=phase,font=('sans-serif',25,'bold')).pack(pady=18)
        tk.Label(root,textvariable=mode,font=('sans-serif',17),fg='#167a55').pack()
        tk.Label(root,textvariable=detail,font=('sans-serif',12),wraplength=640).pack(pady=14)
        tk.Label(root,text='点击本窗口后按键；按钮也可操作。\n窗口操作不是硬件急停，阻塞中的设备操作需先返回。',fg='#666').pack()
        bar=tk.Frame(root);bar.pack(pady=16)
        state={'phase':'CHECK','generation':0,'closing':False}
        held=set(); releases={};buttons={}
        def send(key):
            if not state['closing'] and key in PHASES[state['phase']][1]:
                conn.send(('key',state['generation'],time.monotonic(),key))
        for key,label in [('c','C 继续'),(' ','空格 接管'),('d','D 结束'),('q','Q 排除/跳过'),('x','X 退出')]:
            b=tk.Button(bar,text=label,command=lambda k=key:send(k),takefocus=False)
            b.pack(side='left',padx=3);buttons[key]=b
        def press(event):
            key=' ' if event.keysym=='space' else event.char.lower()
            if not key or key not in ' cdqxh':return
            if key in releases:root.after_cancel(releases.pop(key))
            if key not in held:held.add(key);send(key)
            return 'break'
        def release(event):
            key=' ' if event.keysym=='space' else event.char.lower()
            if key in releases:root.after_cancel(releases[key])
            releases[key]=root.after(60,lambda: (held.discard(key),releases.pop(key,None)))
        root.bind_class('ControlKeys','<KeyPress>',press)
        root.bind_class('ControlKeys','<KeyRelease>',release)
        # Route before Tk Button class bindings (space must never click a focused C button).
        for widget in [root,*buttons.values()]:
            widget.bindtags(('ControlKeys',)+widget.bindtags())
        def close():
            state['closing']=True;conn.send(('close',))
            detail.set('已请求退出，等待当前操作返回；不会通过关窗强制终止设备调用。')
            for b in buttons.values():b.configure(state='disabled')
        root.protocol('WM_DELETE_WINDOW',close)
        def tick():
            try:
                while conn.poll():
                    msg=conn.recv()
                    if msg[0]=='stop':root.destroy();return
                    _,p,g,extra=msg
                    state.update(phase=p,generation=g)
                    phase.set(PHASES[p][0]);mode.set(extra)
                    if p=='READY':root.lift();root.focus_force()
                    if not state['closing']:detail.set(PHASES[p][2])
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
        self.conn.send(('state',phase,self.gate.generation,detail))
    def read_key(self):self._receive();return self.gate.read()
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
