import os
import time
from numpy import float16
from regex import P
import torch
from torch import multiprocessing
from typing import Dict, List


from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.parallel_state import (ensure_model_parallel_initialized,
                                             get_world_group, graph_capture,
                                             init_distributed_environment)
from vllm.utils import update_environment_variables

def worker_fn_wrapper(fn):
    # `multiprocessing.Process` cannot accept environment variables directly
    # so we need to pass the environment variables as arguments
    # and update the environment variables in the function
    def wrapped_fn(env):
        update_environment_variables(env)
        local_rank = os.environ['LOCAL_RANK']
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        init_distributed_environment()
        fn()

    return wrapped_fn

def distributed_run(fn, world_size):
    number_of_processes = world_size
    processes: List[multiprocessing.Process] = []
    for i in range(number_of_processes):
        env: Dict[str, str] = {}
        env['RANK'] = str(i)
        env['LOCAL_RANK'] = str(i)
        env['WORLD_SIZE'] = str(number_of_processes)
        env['LOCAL_WORLD_SIZE'] = str(number_of_processes)
        env['MASTER_ADDR'] = 'localhost'
        env['MASTER_PORT'] = '12345'
        p = multiprocessing.Process(target=fn, args=(env, ))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    for p in processes:
        assert p.exitcode == 0

@worker_fn_wrapper
def test_transfer():
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()
    n = 10
    pynccl_comm = PyNcclCommunicator(get_world_group().cpu_group,
                                     device=get_world_group().device)
    size = 16*2*n
    if pynccl_comm.rank == 0:
        a = torch.ones((16,1024,1024),dtype=torch.float16).cuda(pynccl_comm.rank)
    elif pynccl_comm.rank == 1:
        a = torch.empty((16,1024,1024),dtype=torch.float16).cuda(pynccl_comm.rank)

    if pynccl_comm.rank == 0:
        t = time.time()
        pynccl_comm.send(a,
                         dst=(pynccl_comm.rank + 1) % pynccl_comm.world_size,stream=stream1)
        t = time.time()
        for i in range(n):
            pynccl_comm.send(a,dst=(pynccl_comm.rank + 1) % pynccl_comm.world_size)
    else:
        t = time.time()
        pynccl_comm.recv(a,
                         src=(pynccl_comm.rank - 1) % pynccl_comm.world_size,stream=stream2)
        t = time.time()
        for i in range(n):
            pynccl_comm.recv(a,
                         src=(pynccl_comm.rank - 1) % pynccl_comm.world_size)
    torch.cuda.synchronize()
    now_time = time.time()
    print(f"{pynccl_comm.rank}传输速度为:{size/1024/(now_time-t)}GB/s, 平均耗时为{(now_time-t)/n}\n")

if __name__ == "__main__":
    distributed_run(test_transfer,2) 
    # 大概等于15GB/s , 如果数据传输量很小，速度就会很快变下小到不到1gb/s
    