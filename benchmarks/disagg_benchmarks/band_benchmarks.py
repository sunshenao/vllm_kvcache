import torch
import time

def print_green(text):
    print("\033[92m" + text + "\033[0m")

def benchmark_bandwidth(src_device, dst_device, size_mb=500, loops=10):
    """测试GPU间或与内存间的带宽
    Args:
        src_device: 源设备 (torch.device或"cpu")
        dst_device: 目标设备 (torch.device或"cpu")
        size_mb: 测试数据大小(MB)
        loops: 循环次数
    """
    # 初始化CUDA事件用于精确计时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # 转换为字节数
    bytes_num = int(size_mb * 1024**2)
    element_size = torch.finfo(torch.float32).bits // 8  # 32bit=4bytes
    num_elements = bytes_num // element_size

    # 预分配内存/显存
    if isinstance(src_device, torch.device):
        src_tensor = torch.randn(num_elements, device=src_device)
    else:
        src_tensor = torch.randn(num_elements).pin_memory()

    if isinstance(dst_device, torch.device):
        dst_tensor = torch.empty_like(src_tensor, device=dst_device)
    else:
        dst_tensor = torch.empty_like(src_tensor).cpu().pin_memory()

    # 预热（确保初始化完成）
    torch.cuda.synchronize()
    
    # 执行测试
    total_time_ms = 0
    for _ in range(loops):
        start_event.record()
        stream = torch.cuda.Stream()  # 使用独立流
        with torch.cuda.stream(stream):
            dst_tensor.copy_(src_tensor, non_blocking=True)
        end_event.record()
        torch.cuda.synchronize()  # 等待流完成
        total_time_ms += start_event.elapsed_time(end_event)

    # 计算带宽
    avg_time_ms = total_time_ms / loops
    bandwidth_gbs = (bytes_num * 2) / (avg_time_ms * 1e-3) / 1e9  # 双向带宽
    
    # 输出结果
    direction = ""
    if src_device == "cpu" and isinstance(dst_device, torch.device):
        direction = " (Host->Device)"
    elif isinstance(src_device, torch.device) and dst_device == "cpu":
        direction = " (Device->Host)"
    elif isinstance(src_device, torch.device) and isinstance(dst_device, torch.device):
        if src_device.index == dst_device.index:
            direction = " (D2D Same GPU)"
        else:
            direction = f" (D2D GPU{src_device.index}->GPU{dst_device.index})"
    
    print_green(f"[{direction}] Bandwidth: {bandwidth_gbs:.2f} GB/s")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available!")
        exit()

    # 获取设备信息
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs")

    # 测试设备到设备带宽（同一GPU）
    for i in range(num_gpus):
        device = torch.device(f"cuda:{i}")
        with torch.cuda.device(device):
            benchmark_bandwidth(device, device)
    
    # 测试主机到设备带宽
    benchmark_bandwidth("cpu", torch.device("cuda:0"))
    
    # 测试设备到主机带宽
    benchmark_bandwidth(torch.device("cuda:0"), "cpu")
    
    # 测试跨GPU带宽（如果有多GPU）
    if num_gpus >= 2:
        for i in range(num_gpus):
            for j in range(num_gpus):
                if i != j:
                    src = torch.device(f"cuda:{i}")
                    dst = torch.device(f"cuda:{j}")
                    # 检查P2P访问是否支持
                    if torch.cuda.can_device_access_peer(src, dst):
                        benchmark_bandwidth(src, dst)
                    else:
                        print(f"P2P access not supported between GPU{i} and GPU{j}")

                        