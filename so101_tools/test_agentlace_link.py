#!/usr/bin/env python
"""隔离测试 agentlace 的 TrainerServer <-> TrainerClient 数据通路。

目的：B0.2 里 learner 的 replay_buffer 一直是空的。这个脚本把
「网络传输」从「RLPD 训练逻辑」里剥离出来单独验证，
以免在庞大的 train_rlpd.py 里瞎猜。

预期：server 端 buffer 长度从 0 增长到 20，并打印收到的数据。
"""

import multiprocessing as mp
import os
import sys
import time

import numpy as np

from agentlace.trainer import TrainerClient, TrainerServer, TrainerConfig
from agentlace.data.data_store import QueuedDataStore


class ListStore:
    """最简 server 端数据存储：只记录收到了什么。

    注意：agentlace 的 server 回调调用的是 batch_insert（不是 insert），
    且要求能返回 __len__（server 用它算 last_update_id）。
    """

    def __init__(self):
        self.items = []

    def insert(self, data):
        self.items.append(data)

    def batch_insert(self, data):
        # server 端传进来的是 list
        if isinstance(data, (list, tuple)):
            self.items.extend(data)
        else:
            self.items.append(data)

    def __len__(self):
        return len(self.items)


def server_main(port, bcast, result_q):
    cfg = TrainerConfig(port_number=port, broadcast_port=bcast)
    server = TrainerServer(cfg, request_callback=lambda t, p: {})
    store = ListStore()
    server.register_data_store("actor_env", store)
    server.start(threaded=True)
    print(f"[server] listening {port}/{bcast}", flush=True)

    # 等 client 送数据
    t0 = time.time()
    while len(store) < 20 and time.time() - t0 < 40:
        time.sleep(0.5)
    print(f"[server] 收到 {len(store)} 条", flush=True)
    result_q.put(len(store))
    time.sleep(1)


def client_main(port, bcast, ready_evt):
    cfg = TrainerConfig(port_number=port, broadcast_port=bcast)
    ds = QueuedDataStore(1000)
    client = TrainerClient(
        "actor_env",
        "127.0.0.1",
        cfg,
        data_stores={"actor_env": ds, "actor_env_intvn": QueuedDataStore(1000)},
        wait_for_server=True,
        timeout_ms=3000,
    )
    print("[client] connected", flush=True)
    ready_evt.set()

    for i in range(20):
        ds.insert({"observations": {"state": np.zeros(4)}, "rewards": i, "step": i})
    ok = client.update()
    print(f"[client] update() -> {ok}", flush=True)
    print(f"[client] data_store len = {len(ds)}", flush=True)
    time.sleep(2)


if __name__ == "__main__":
    port, bcast = 5699, 5700
    q = mp.Queue()
    evt = mp.Event()

    ps = mp.Process(target=server_main, args=(port, bcast, q))
    pc = mp.Process(target=client_main, args=(port, bcast, evt))
    ps.start()
    time.sleep(3)
    pc.start()

    ps.join(timeout=60)
    pc.join(timeout=10)

    got = None
    while not q.empty():
        got = q.get()
    print(f"\n结果: server 收到 {got} 条" + ("  ✅ 通路正常" if got else "  ❌ 数据没送达"))
    sys.exit(0 if got else 1)
