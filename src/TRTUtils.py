# Based on https://github.com/NVIDIA/object-detection-tensorrt-example/blob/master/SSD_Model/utils/inference.py

import sys
import tensorrt as trt
from PIL import Image
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from typing import Tuple, Union
import logging
import time


class TrtInfer():
    '''Class for inference tensorrt models with a static batch size'''
    def __init__(self, path_to_model: str):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        trt_runtime = trt.Runtime(TRT_LOGGER)
        self.engine = load_engine(trt_runtime, path_to_model)
        self.inputs, self.outputs, self.bindings, self.stream = TrtInfer.allocate_buffers(self.engine)
        self.context =  self.engine.create_execution_context()


    def inference(self, input: np.ndarray):
        np.copyto(self.inputs[0].host, input.flatten())
        output = do_inference(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)
        return output


    def inference_with_time(self, input: np.ndarray):
        start_time = time.time()
        np.copyto(self.inputs[0].host, input.flatten())
        start_gpu_time = time.time()
        output = do_inference(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)
        end_gpu_time = time.time()
        return output, start_time, start_gpu_time, end_gpu_time


    @staticmethod
    def allocate_buffers(engine):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream


class TrtDynamicInfer():
    '''Class for inference tensorrt models with a dynamic or static batch size'''
    def __init__(self, path_to_model: str):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        trt_runtime = trt.Runtime(TRT_LOGGER)
        self.engine = load_engine(trt_runtime, path_to_model)
        self.context =  self.engine.create_execution_context()


    def inference(self, input: np.ndarray):
        self.context.set_binding_shape(0, input.shape)
        inputs, outputs, bindings, stream = TrtDynamicInfer.allocate_buffers(self.engine, current_batch_size=input.shape[0])
        np.copyto(inputs[0].host, input.flatten())
        output = do_inference(self.context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        return output


    def inference_with_time(self, input: np.ndarray):
        start_time = time.time()
        self.context.set_binding_shape(0, input.shape)
        inputs, outputs, bindings, stream = TrtDynamicInfer.allocate_buffers(self.engine, current_batch_size=input.shape[0])
        np.copyto(inputs[0].host, input.flatten())
        start_gpu_time = time.time()
        output = do_inference(self.context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        end_gpu_time = time.time()
        return output, start_time, start_gpu_time, end_gpu_time


    @staticmethod
    def allocate_buffers(engine, current_batch_size: int):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        for binding in engine:
            shape = engine.get_binding_shape(binding)
            if shape[0] == -1:
                # if the source trt model has a dynamic batch size (shape=(-1, C, H, W))
                # we need to allocate memory with using batch size of received data
                size = trt.volume(shape) * (-current_batch_size)
            else:
                # if the source trt model has a static batch size (shape=(B, C, H, W))
                # we don't need to change anything
                size = trt.volume(engine.get_binding_shape(binding))

            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    flag = context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def build_engine_onnx(input_onnx: Union[str, bytes], force_fp16: bool, min_shape: tuple=None, opt_shape: tuple=None, max_shape: tuple=None):
    '''
    Fuilds TensorRT engine from provided ONNX file
    Input parameters:
    input_onnx: Union[str, bytes - serialized ONNX model
    force_fp16: bool - force use of FP16 precision, even if device doesn't support it. Be careful
    min_shape: tuple - min data shape
    opt_shape: tuple - optimal data shape
    max_shape: tuple - max data shape
    Return: TensorRT engine
    '''

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(explicit_batch) as network, \
            builder.create_builder_config() as config, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:

        if force_fp16 is True:
            logging.info('Building TensorRT engine with FP16 support.')
            has_fp16 = builder.platform_has_fast_fp16
            if not has_fp16:
                logging.warning('Builder report no fast FP16 support. Performance drop expected')
            config.set_flag(trt.BuilderFlag.FP16)
            config.set_flag(trt.BuilderFlag.STRICT_TYPES)

        config.max_workspace_size = 1 << 22
        
        if not parser.parse(input_onnx):
            print('ERROR: Failed to parse the ONNX')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            sys.exit(1)


        # Nvidia developers dont recommend creating optimization profiles for
        # models with static dimentions
        if ((min_shape!=None) and (opt_shape!=None) and (max_shape!=None)):
            profile = builder.create_optimization_profile()

            if min_shape[0] != max_shape[0]:
                logging.warning('Dynamic batch size is used. Ensure your inference code supports it')
            
            print(f"TRT model min shape = {min_shape}")
            print(f"TRT model opt shape = {opt_shape}")
            print(f"TRT model max shape = {max_shape}")

            profile.set_shape('input' , min_shape , opt_shape, max_shape)
            config.add_optimization_profile(profile)

        return builder.build_engine(network, config)
        

def save_engine(engine, engine_dest_path):
    assert not isinstance(engine, type(None))
    with open(engine_dest_path, "wb") as f:
        f.write(engine.serialize())


def load_engine(trt_runtime, engine_path):
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine


# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def load_data(pagelocked_buffer, input_data):
    return np.copyto(pagelocked_buffer, input_data)