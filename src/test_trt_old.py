# Script for testing models with only one input and output
# Script generates invalid output with fp16 input

import tensorrt as trt
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda 
import time
from pycuda.tools import DeviceMemoryPool as DMP
from pycuda.tools import PageLockedMemoryPool as PMP
import torch
import argparse

class TrtInfer():
    def __init__(self, path_to_model: str):
        self.device_pool=DMP()
        self.host_pool = PMP()
        trt_path = path_to_model
        #self.input_size = 384
        self.engine = None


        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)
        with open(trt_path, 'rb') as f:
            engine_bytes = f.read()
            self.engine = runtime.deserialize_cuda_engine(engine_bytes)

        self.context = self.engine.create_execution_context()
        print('context done')

        # host cpu mem
        h_in_size = trt.volume(self.engine.get_binding_shape(0))
        h_out_size = trt.volume(self.engine.get_binding_shape(1))
        h_in_dtype = trt.nptype(self.engine.get_binding_dtype(0))
        h_out_dtype = trt.nptype(self.engine.get_binding_dtype(1))

        #allocate host mem
        in_cpu = cuda.pagelocked_empty(h_in_size, h_in_dtype)
        out_cpu = cuda.pagelocked_empty(h_out_size, h_out_dtype)
        #in_cpu = host_pool.allocate([h_in_size], h_in_dtype)
        #out_cpu = host_pool.allocate([h_out_size], h_out_dtype)
        
        # allocate gpu mem
        in_gpu = cuda.mem_alloc(in_cpu.nbytes)
        out_gpu = cuda.mem_alloc(out_cpu.nbytes)
        #in_gpu = device_pool.allocate(in_cpu.nbytes)
        #out_gpu = device_pool.allocate(out_cpu.nbytes)
        stream = cuda.Stream()
        print('alloc done')

        self.in_cpu = in_cpu
        self.out_cpu = out_cpu
        self.in_gpu = in_gpu
        self.out_gpu = out_gpu
        self.stream = stream


    def inference(self, inputs):
        # async version
        inputs = inputs.reshape(-1)
        cuda.memcpy_htod_async(self.in_gpu, inputs, self.stream)
        #context.execute_async(1, [int(in_gpu), int(out_gpu)], stream.handle, None)
        self.context.execute_async_v2(bindings=[int(self.in_gpu), int(self.out_gpu)], stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.out_cpu, self.out_gpu, self.stream)
        self.stream.synchronize()

        return self.out_cpu


def benchmark(session, input_shape=(1, 3, 128, 128), dtype='fp32', nruns=10000):

    timings = []
    input_data = torch.randn(input_shape)
    if dtype=='fp16':
        input_data = input_data.half()
        #input_data = (input_data.astype(np.float32))
    input_data = input_data.numpy()

    for i in range(1, nruns+1):
        start_time = time.time()
        res = session.inference(input_data)
        #torch.cuda.synchronize()
        end_time = time.time()
        timings.append(end_time - start_time)
        if i%1000==0:
            print('Iteration %d/%d, ave batch time %.2f ms'%(i, nruns, np.mean(timings)*1000))

    print("Input shape:", input_data.shape)
    print("Output features size:", res.shape)
    print('Average batch time: %.2f ms'%(np.mean(timings)*1000))


if __name__ == "__main__":

  parser = argparse.ArgumentParser(description="Test TRT")
  parser.add_argument("-n", dest="name", help="model name", required=True, type=str)
  parser.add_argument("-bs", dest="batchsize", help="batch size", required=True, type=int)
  parser.add_argument("-is", dest="image_size", help="size of image", required=True, type=int)
  parser.add_argument("-prec", dest="precision", help="fp16 or fp32", required=True, choices=['fp16', 'fp32'])
  args = parser.parse_args()
  path_to_model = f'../models/trt/{args.name}'
  trt_infer = TrtInfer(path_to_model=path_to_model)
  benchmark(trt_infer, input_shape=(args.batchsize, 3, args.image_size, args.image_size), dtype=args.precision, nruns=10000)