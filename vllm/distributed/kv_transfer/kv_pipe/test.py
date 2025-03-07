
import asyncio
import threading
import time
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Optional, Tuple

from httpx import stream
import torch

from vllm.config import KVTransferConfig
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.kv_transfer.kv_pipe.base import KVPipeBase
from vllm.distributed.utils import StatelessProcessGroup
from vllm.logger import init_logger

logger = init_logger(__name__)


class BrokenPipeException(Exception):

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


Metadata = Dict[str, Optional[torch.Tensor]]


class LayerPipe(KVPipeBase):

    METADATA_LENGTH = 16
    MAX_TENSOR_DIMENSIONS = 14
    METADATA_DTYPE = torch.int64

    def __init__(self,
                 local_rank: int,
                 config: KVTransferConfig,
                 device: Optional[str] = None,
                 port_offset: int = 0,
                 stream: torch.cuda.Stream = None):
        self.config = config
        self.local_rank = local_rank
        self.kv_rank = self.config.kv_rank
        self.stream = stream
        
        self.send_stream: torch.cuda.Stream = torch.cuda.Stream()
        self.recv_stream: torch.cuda.Stream = torch.cuda.Stream()
        self.kv_parallel_size = self.config.kv_parallel_size
        if device is None:
            self.device = self._select_device(self.config.kv_buffer_device)
        else:
            self.device = self._select_device(device)

        self.group = StatelessProcessGroup.create(
            host=self.config.kv_ip,
            port=self.config.kv_port + port_offset,
            rank=self.kv_rank,
            world_size=self.kv_parallel_size,
        )
        self.group.barrier()
        impl = self._get_device_send_recv_impl(self.group)
        self.device_send_func, self.device_recv_func = impl
        
        self.target_rank_for_send = (self.kv_rank + 1) % self.kv_parallel_size
        self.target_rank_for_recv = (self.kv_rank - 1) % self.kv_parallel_size

        self.transport_thread: Optional[ThreadPoolExecutor] = None
        self.buffer_size = 0
        self.buffer_size_lock = threading.Lock()
        self.buffer_size_thresh = self.config.kv_buffer_size

        

    def _get_device_send_recv_impl(
        self, group: StatelessProcessGroup
    ) -> Tuple[Callable[[torch.Tensor, int], None], Callable[
        [torch.Tensor, int], None]]:

        send: Callable[[torch.Tensor, int], None]
        recv: Callable[[torch.Tensor, int], None]
        if self.device.type == "cuda":
 
            comm = PyNcclCommunicator(group, device=self.local_rank)
            comm.disabled = False

            
            # send = functools.partial(comm.send,stream=self.stream)
            # recv = functools.partial(comm.recv,stream=self.stream)
            # send, recv = comm.send, comm.recv  # type: ignore
            send, recv = comm.async_send, comm.async_recv 
            # print('-='*19)
        else:
            
            send = group.send_obj

            def my_recv(x, src):
                x[...] = group.recv_obj(src)

            recv = my_recv

        return send, recv

    def _select_device(self, device: str):
        logger.info("Selecting device: %s", device)
        if device == "cuda":
            return torch.device(f"cuda:{self.local_rank}")
        else:
            return torch.device("cpu")

    def _make_metadata(self, tensor: Optional[torch.Tensor]) -> Metadata:

        if tensor is None:
            return {"dtype": None, "shape": None}
        else:
            return {"dtype": tensor.dtype, "shape": tensor.shape}

    def _prepare_recv_buffer(self, metadata: Metadata) -> torch.Tensor:

        return torch.empty(metadata["shape"],
                           dtype=metadata["dtype"],
                           device=self.device)

    def _send_metadata(self, metadata: Metadata):

        self.group.send_obj(metadata, self.target_rank_for_send)

    def _recv_metadata(self) -> Metadata:

        return self.group.recv_obj(self.target_rank_for_recv)

    def _send_impl(self, tensor: Optional[torch.Tensor]) -> None:
        # with torch.cuda.stream(self.send_stream):
        metadata = self._make_metadata(tensor)
        self._send_metadata(metadata)
        if tensor is not None:
            self.device_send_func(tensor.to(self.device),
                                self.target_rank_for_send)

    def _recv_impl(self) -> Optional[torch.Tensor]:
        # with torch.cuda.stream(self.send_stream):
        metadata = self._recv_metadata()
        if metadata["dtype"] is None:
            return None
        buffer = self._prepare_recv_buffer(metadata)
        self.device_recv_func(buffer, self.target_rank_for_recv)

        return buffer

    def send_tensor_wrapper(self, tensor: Optional[torch.Tensor],
                            tensor_size: int) -> None:

        try:
            self._send_impl(tensor)

            with self.buffer_size_lock:
                self.buffer_size -= tensor_size
        except Exception as e:
            logger.error("[rank%d]: Exception when trying to send %s, msg: %s",
                         torch.distributed.get_rank(), str(tensor), str(e))
            import traceback
            traceback.print_exc()

    def block_if_full(self):

        while self.buffer_size > self.buffer_size_thresh:
            logger.debug("KV cache transfer pipe is full. Waiting...")
            time.sleep(0.01)

    def send_tensor(self, tensor: Optional[torch.Tensor]) -> None:

        if self.transport_thread is None:
            self.transport_thread = ThreadPoolExecutor(max_workers=1)

        if tensor is not None:
            tensor_size = tensor.element_size() * tensor.numel()
        else:
            tensor_size = 0

        self.block_if_full()

        with self.buffer_size_lock:
            self.buffer_size += tensor_size

        self.transport_thread.submit(self.send_tensor_wrapper, tensor,
                                     tensor_size)

    def recv_tensor(self) -> Optional[torch.Tensor]:

        if self.transport_thread is None:
            self.transport_thread = ThreadPoolExecutor(max_workers=1)

        future = self.transport_thread.submit(self._recv_impl)

        try:
            tensor = future.result()
        except Exception as e:
            logger.error("Encountering exception in KV receiving thread")
            logger.error("%s", e)
            logger.error("My device: %s", self.device)
            import traceback
            traceback.print_exc()
            raise e

        return tensor

    def close(self):

        if hasattr(self,
                   "transport_thread") and self.transport_thread is not None:
            self.transport_thread.shutdown()




    async def async_send_tensor(self, tensor: Optional[torch.Tensor]) -> None:
        # if self.transport_thread is None:
        #     self.transport_thread = ThreadPoolExecutor(max_workers=1)

        if tensor is not None:
            tensor_size = tensor.element_size() * tensor.numel()
        else:
            tensor_size = 0

        self.block_if_full()

        with self.buffer_size_lock:
            self.buffer_size += tensor_size
        await self.async_send_tensor_wrapper(tensor, tensor_size)

        # await asyncio.get_event_loop().run_in_executor(
        #     self.transport_thread, self.send_tensor_wrapper, tensor, tensor_size)

    async def async_recv_tensor(self) -> Optional[torch.Tensor]:
        # if self.transport_thread is None:
        #     self.transport_thread = ThreadPoolExecutor(max_workers=1)

        # future = await asyncio.get_event_loop().run_in_executor(
        #     self.transport_thread, self._recv_impl)

        try:
            # tensor = future.result()
            tensor = await self._async_recv_impl()
        except Exception as e:
            logger.error("Encountering exception in KV receiving thread")
            logger.error("%s", e)
            logger.error("My device: %s", self.device)
            import traceback
            traceback.print_exc()
            raise e

        return tensor
    

    async def _async_send_metadata(self, metadata: Metadata):

        self.group.send_obj(metadata, self.target_rank_for_send)

    async def _async_recv_metadata(self) -> Metadata:

        return self.group.recv_obj(self.target_rank_for_recv)

    async def _async_send_impl(self, tensor: Optional[torch.Tensor]) -> None:
        # with torch.cuda.stream(self.send_stream):
        metadata = self._make_metadata(tensor)
        await self._async_send_metadata(metadata)
        if tensor is not None:
            await self.device_send_func(tensor.to(self.device),
                                self.target_rank_for_send)

    async def _async_recv_impl(self) -> Optional[torch.Tensor]:
        # with torch.cuda.stream(self.send_stream):
        metadata = await self._async_recv_metadata()
        if metadata["dtype"] is None:
            return None
        buffer = self._prepare_recv_buffer(metadata)
        await self.device_recv_func(buffer, self.target_rank_for_recv)

        return buffer

    async def async_send_tensor_wrapper(self, tensor: Optional[torch.Tensor],
                            tensor_size: int) -> None:

        try:
            await self._async_send_impl(tensor)

            with self.buffer_size_lock:
                self.buffer_size -= tensor_size
        except Exception as e:
            logger.error("[rank%d]: Exception when trying to send %s, msg: %s",
                         torch.distributed.get_rank(), str(tensor), str(e))
            import traceback
            traceback.print_exc()