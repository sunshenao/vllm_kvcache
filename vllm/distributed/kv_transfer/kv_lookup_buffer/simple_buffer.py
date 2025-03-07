# SPDX-License-Identifier: Apache-2.0
"""
    Implements a distributed key-value (KV) cache transfer mechanism.

    Key Features:
    - Distributed KV cache transmission using PyNccl pipes.
    - Non-blocking `insert`, blocking `drop_select`.
    - Use CPU signal pipe to avoid racing condition
    - Handles buffer size constraints and provide backpressure mechanism to 
      stop the prefill instance when the decode instance is slow.
"""
import asyncio
import threading
from collections import deque
import time
from typing import Deque, List, Optional, Union

import torch

from vllm.distributed.kv_transfer.kv_lookup_buffer.base import (
    KVLookupBufferBase)
from vllm.distributed.kv_transfer.kv_pipe.base import KVPipeBase
from vllm.logger import init_logger

logger = init_logger(__name__)


class SimpleBuffer(KVLookupBufferBase):

    def __init__(self, signal_pipe: KVPipeBase, data_pipe: KVPipeBase,
                 buffer_size_thresh: float):
        """
        signal_pipe: on CPU

        NOTE: on-device recv will block all threads in the process, making the
        KV cache producer unable to listen to new request while transmitting
        KV cache. Luckily CPU recv only blocks the current thread so we use
        CPU recv to listen to new request.

        data_pipe: on device (e.g. GPU)
        """

        self.buffer: Deque[List[torch.Tensor]] = deque()

        self.buffer_size = 0
        self.buffer_size_threshold = buffer_size_thresh
        self.buffer_cv = threading.Condition()
        self.signal_pipe = signal_pipe
        self.data_pipe = data_pipe
        self.request_handling_thread: Optional[threading.Thread] = None

        self.normal_signal = torch.tensor([0], device="cpu")
        self.end_signal = None
  
        # self.signal_pipe = signal_pipe
        # self.data_pipe = data_pipe
        # self.stream_a = torch.cuda.Stream()
        # self.stream_b = torch.cuda.Stream()
        # self.event_a = torch.cuda.Event()
        # self.event_b = torch.cuda.Event()

    def _matches(self, tokens_roi_sender: List[torch.Tensor],
                 tokens_roi_recver: List[torch.Tensor]):
        # b = time.time()
        # tokens_roi_sender: tokens and roi of the producer (in the buffer)
        # tokens_roi_recver: tokens and roi of the consumer (query)

        tokens_sender = tokens_roi_sender[0]
        tokens_recver = tokens_roi_recver[0]
        roi_sender = tokens_roi_sender[1]
        roi_recver = tokens_roi_recver[1]

        if tokens_recver is None:
            # consumer sends an empty request
            # semantics: DROP SELECT * LIMIT 1
            # so any of the data in the buffer can be drop-selected
            return True

        # Assuming that roi is a binary mask on tokens
        # c = time.time()
        #with torch.cuda.stream(self.stream_a):
        tokens_sender = tokens_sender[roi_sender]
        #with torch.cuda.stream(self.stream_b):
        tokens_recver = tokens_recver[roi_recver]
        torch.cuda.synchronize()
        # d = time.time()
        # simple common prefix matching
        min_length = min(len(tokens_sender), len(tokens_recver))
        # a = time.time()
        if torch.allclose(tokens_sender[:min_length],
                          tokens_recver[:min_length]):
            # print('match中的匹配时间',time.time()-a,min_length)
            # print('match总时间为: ',time.time() - b)
            # print('match赋值的时间为,',d-c)
            return min_length

        return 0

    async def _send_tensor_and_dec_size(self,
                                  tensor: Optional[torch.Tensor]) -> None:

        assert tensor is not None, "Use self.data_pipe.send(None) instead"
        self.buffer_size -= tensor.element_size() * tensor.numel()
        if tensor.dtype == torch.bool:
            tensor = tensor.float()
        await self.data_pipe.send_tensor(tensor)

    def _get_element_size(self, data: Optional[Union[List, torch.Tensor]]):

        if isinstance(data, torch.Tensor):
            return data.element_size() * data.numel()
        if not data:
            # cannot perform `not data` on a tensor
            # so this check needs to go after the check above
            return 0

        raise AssertionError(f"Unknown data type {type(data)}")

    def _add_to_buffer(self, input_tokens: torch.Tensor, roi: torch.Tensor,
                       key: torch.Tensor, value: torch.Tensor,
                       hidden: torch.Tensor):

        if isinstance(input_tokens, torch.Tensor):
            input_tokens = input_tokens.clone()
        if isinstance(roi, torch.Tensor):
            roi = roi.clone()
        if isinstance(key, torch.Tensor):
            key = key.clone()
        if isinstance(value, torch.Tensor):
            value = value.clone()
        if isinstance(hidden, torch.Tensor):
            hidden = hidden.clone()

        buffer_item = [input_tokens, roi, key, value, hidden]
        data_size = sum([self._get_element_size(data) for data in buffer_item])

        with self.buffer_cv:
            if self.buffer_size + data_size > self.buffer_size_threshold:
                # log outside the while loop to avoid this message being logged
                # repeatedly.
                logger.debug("KV transfer buffer is full. Handling...")
                while self.buffer_size + data_size > self.buffer_size_threshold:
                    self.buffer_cv.wait()

            self.buffer_size += data_size
            self.buffer.append(buffer_item)
            self.buffer_cv.notify()

    def _is_end_signal(self, signal):
        return signal is None

    async def drop_select_handler(self):

        try:

            while True:
                signal = await self.signal_pipe.recv_tensor()
                if self._is_end_signal(signal):
                    logger.info("Received end signal!")
                    break

                input_tokens = await self.data_pipe.recv_tensor()

                roi = await self.data_pipe.recv_tensor()

                input_tokens = input_tokens.contiguous()
                roi = roi.contiguous()
                assert roi is not None, "Please provide the roi when sending "\
                    "drop-select request"
                roi = (roi > 0.5)
                tokens_roi_recver = [input_tokens, roi]

                def is_buffer_available(
                    tokens_roi_recver: List[torch.Tensor], ) -> bool:
                    # perform input tokens and roi matching
                    # FIXME: this matching is O(n), ideally it should be O(1)
                    # but this buffer size won't (and shouldn't) be too large so
                    # the fix is not urgent.
                    for _ in range(len(self.buffer)):
                        if self._matches(self.buffer[0],
                                         tokens_roi_recver) > 0:
                            return True
                        # rotate the element we just accessed to the end
                        self.buffer.rotate(-1)
                    return False

                with self.buffer_cv:
                    while not is_buffer_available(tokens_roi_recver):
                        logger.debug(
                            "KV transfer buffer is not available. Waiting...")
                        self.buffer_cv.wait()
                    # need to clone the tensor
                    # in case the tensor is freed before sending finishes
                    matched_item = self.buffer.popleft()
                    for tensor in matched_item:
                        await self._send_tensor_and_dec_size(tensor)
                    self.buffer_cv.notify()

        except RuntimeError as e:
            if 'Connection closed by peer' not in str(e):
                raise e

        logger.debug("Closing drop_select_handler")

    async def async_drop_select(
            self, input_tokens: Optional[torch.Tensor],
            roi: Optional[torch.Tensor]) -> List[Optional[torch.Tensor]]:

        assert self.request_handling_thread is None, \
            "drop_select should be called by the KV cache consumer "\
            "(e.g. the decode vLLM instance)"

        if isinstance(input_tokens, torch.Tensor):
            input_tokens = input_tokens.clone()
        if isinstance(roi, torch.Tensor):
            roi = roi.clone().float()

        await self.signal_pipe.send_tensor(self.normal_signal)
        await self.data_pipe.send_tensor(input_tokens)
        await self.data_pipe.send_tensor(roi)

        input_tokens = await self.data_pipe.recv_tensor()
        roi = await self.data_pipe.recv_tensor()
        if roi is not None:
            # convert from float tensor to bool tensor
            # as PyNccl does not support sending bool tensor
            roi = (roi > 0.5)
        key = await self.data_pipe.recv_tensor()
        value = await self.data_pipe.recv_tensor()
        hidden = await self.data_pipe.recv_tensor()

        return [input_tokens, roi, key, value, hidden]

    def drop_select_handler_working(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.drop_select_handler())
        loop.close()

    async def async_insert(self, input_tokens: torch.Tensor, roi: torch.Tensor,
               key: torch.Tensor, value: torch.Tensor,
               hidden: torch.Tensor) -> None:

        self._add_to_buffer(input_tokens, roi, key, value, hidden)

        # when calling the insert, the current process is a sender
        # need to launch the request handler and start listening to request.
        if self.request_handling_thread is None:
            self.request_handling_thread = threading.Thread(
                target=self.drop_select_handler_working)
            self.request_handling_thread.start()
    
    def close(self):

        if hasattr(self, "request_handling_thread"
                   ) and self.request_handling_thread is not None:
            self.request_handling_thread.join()

        else:
            # TODO: have a explicit close signal and have a explicit way to
            # check if it's requester
            self.signal_pipe.send_tensor(self.end_signal)


    # def drop_select_handler(self):

    #     try:

    #         while True:
    #             signal = self.signal_pipe.recv_tensor()
    #             if self._is_end_signal(signal):
    #                 logger.info("Received end signal!")
    #                 break

    #             input_tokens = self.data_pipe.recv_tensor().contiguous()

    #             roi = self.data_pipe.recv_tensor()
    #             assert roi is not None, "Please provide the roi when sending "\
    #                 "drop-select request"
    #             roi = (roi > 0.5)
    #             tokens_roi_recver = [input_tokens, roi]

    #             def is_buffer_available(
    #                 tokens_roi_recver: List[torch.Tensor], ) -> bool:
    #                 # perform input tokens and roi matching
    #                 # FIXME: this matching is O(n), ideally it should be O(1)
    #                 # but this buffer size won't (and shouldn't) be too large so
    #                 # the fix is not urgent.
    #                 b = time.time()
    #                 for i in range(len(self.buffer)):
    #                     if self._matches(self.buffer[0],
    #                                      tokens_roi_recver) > 0:
    #                         print(f"is_buffer_available时间『{i}』:",time.time() - b)
    #                         return True
    #                     # rotate the element we just accessed to the end
    #                     self.buffer.rotate(-1)
    #                 return False
    #             a = time.time()
    #             with self.buffer_cv:
    #                 f = time.time()
    #                 while not is_buffer_available(tokens_roi_recver):
    #                     print('-='*12)
    #                     logger.debug(
    #                         "KV transfer buffer is not available. Waiting...")
    #                     # time.sleep(0.003)
    #                     self.buffer_cv.wait()
    #                 g = time.time()
    #                 print('while循环时间为:,', g- f)
    #                 # need to clone the tensor
    #                 # in case the tensor is freed before sending finishes
    #                 matched_item = self.buffer.popleft()
                    
    #                 print("drop_select_handler时间:",time.time() - f)
    #                 for tensor in matched_item:
    #                     self._send_tensor_and_dec_size(tensor)
    #                 self.buffer_cv.notify()

    #     except RuntimeError as e:
    #         if 'Connection closed by peer' not in str(e):
    #             raise e

    #     logger.debug("Closing drop_select_handler")

    # def drop_select(
    #         self, input_tokens: Optional[torch.Tensor],
    #         roi: Optional[torch.Tensor]) -> List[Optional[torch.Tensor]]:

    #     assert self.request_handling_thread is None, \
    #         "drop_select should be called by the KV cache consumer "\
    #         "(e.g. the decode vLLM instance)"

    #     if isinstance(input_tokens, torch.Tensor):
    #         input_tokens = input_tokens.clone()
    #     if isinstance(roi, torch.Tensor):
    #         roi = roi.clone().float()

    #     a = time.time()
    #     self.signal_pipe.send_tensor(self.normal_signal)
    #     b = time.time()
    #     self.data_pipe.send_tensor(input_tokens)
    #     c = time.time()
    #     self.data_pipe.send_tensor(roi)
    #     d = time.time()
    #     print("send时间：",[b-a,c-b,d-c])
    #     # self.data_pipe.send_tensor([])

    #     a = time.time()
    #     input_tokens = self.data_pipe.recv_tensor() # 这个速度慢
    #     b = time.time()
    #     roi = self.data_pipe.recv_tensor()
    #     c = time.time()
    #     if roi is not None:
    #         # convert from float tensor to bool tensor
    #         # as PyNccl does not support sending bool tensor
    #         roi = (roi > 0.5)
    #     key = self.data_pipe.recv_tensor()
    #     d = time.time()
    #     value = self.data_pipe.recv_tensor()
    #     e = time.time()
    #     hidden = self.data_pipe.recv_tensor()
    #     g = time.time()
    #     print("recv时间：",[b-a,c-b,d-c,e-d,g-e])

    #     # print([input_tokens, roi, key, value, hidden])
    #     # print(self.normal_signal)

    #     return [input_tokens, roi, key, value, hidden]