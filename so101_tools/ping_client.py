import zmq
import time

# 替换为服务端的实际局域网 IP
SERVER_IP = "127.0.0.1" 
PORT = 5555
TEST_COUNT = 100  # 测试次数

def main():
    context = zmq.Context()
    # 使用 REQ (Request) 模式
    socket = context.socket(zmq.REQ)
    
    print(f"正在连接到服务器 tcp://{SERVER_IP}:{PORT}...")
    socket.connect(f"tcp://{SERVER_IP}:{PORT}")
    
    print("开始 Ping-Pong 延迟测试...\n")
    
    latencies = []
    
    try:
        for i in range(TEST_COUNT):
            # 记录发送前的时间
            start_time = time.perf_counter()
            
            # 发送 PING
            socket.send_string("PING")
            
            # 阻塞等待回复
            reply = socket.recv_string()
            
            # 记录接收后的时间
            end_time = time.perf_counter()
            
            if reply == "PONG":
                # 计算往返耗时 (毫秒)
                rtt = (end_time - start_time) * 1000
                latencies.append(rtt)
                print(f"Seq {i+1:03d}: 收到 PONG, 往返延迟 = {rtt:.2f} ms")
            else:
                print(f"Seq {i+1:03d}: 收到未知回复 {reply}")
                
            # 短暂休眠，模拟真实控制频率（例如 30Hz 大约是 0.033 秒）
            time.sleep(0.033)
            
    except KeyboardInterrupt:
        print("\n测试被手动中断。")
    finally:
        if latencies:
            print("\n--- Ping-Pong 测试统计 ---")
            print(f"发送数据包: {len(latencies)}")
            print(f"最小延迟: {min(latencies):.2f} ms")
            print(f"最大延迟: {max(latencies):.2f} ms")
            print(f"平均延迟: {sum(latencies)/len(latencies):.2f} ms")
            
        socket.close()
        context.term()

if __name__ == "__main__":
    main()