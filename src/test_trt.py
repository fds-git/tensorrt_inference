import numpy as np
from TRTUtils import TrtInfer, TrtDynamicInfer
import tensorrt as trt 
import time
import argparse
from common_utils import image_shape_type
import logging

# !!! In inference time you should put fp32 data into fp16 optimized model
# !!! othervise data proccesing time will be increased

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test TRT")
    parser.add_argument("-n", dest="name", help="model name", required=True, type=str)
    parser.add_argument("-bs", dest="batch_size", help='batch size', required=False, type=int, default=None)
    parser.add_argument("-prec", dest="precision", help="fp16 or fp32", required=True, choices=['fp16', 'fp32'])
    parser.add_argument("-infer", dest="inference_type", help="stat or dinstat", required=True, choices=['stat', 'dinstat'])
    args = parser.parse_args()

    path_to_model = f'../models/trt/{args.name}'

    if args.inference_type == 'stat':
        logging.warning('This inference type is able to work with only static models. Ensure, your model is static')
        trt_infer = TrtInfer(path_to_model=path_to_model)
    else:
        logging.info('This inference type is able to work with static and dynamic models')
        trt_infer = TrtDynamicInfer(path_to_model=path_to_model)
    
    input_shape = trt_infer.engine.get_binding_shape(index=0)
    # if model has a dynamic batch size
    if input_shape[0] == -1:
        if args.batch_size == None:
            raise Exception("This model has a dynamic batch size. You must set this value")
        else:
            input_shape[0] = args.batch_size
    # if models has a static batch size
    else:
        if args.batch_size != None:
            logging.warning(f'This model has a static batch size. Received batch size value will be ignored. Model\'s batch size is {input_shape[0]}')
    
    input_ = np.random.rand(*input_shape)
    dtype = np.float16 if args.precision == 'fp16' else np.float32
    input_ = input_.astype(dtype)

    cpu_timings, gpu_timings, all_timings = [], [], []
    print("*"*50)
    nruns = 10000
    for i in range(1, nruns+1):
        output, start_time, start_gpu_time, end_gpu_time = trt_infer.inference_with_time(input_)

        cpu_timings.append(start_gpu_time - start_time)
        gpu_timings.append(end_gpu_time - start_gpu_time)
        all_timings.append(end_gpu_time - start_time)

        if i%1000==0:
            print(f'Iteration {i}/{nruns}, ave batch time: CPU: {np.mean(cpu_timings)*1000:.3}, GPU: {np.mean(gpu_timings)*1000:.3}, ALL: {np.mean(all_timings)*1000:.3}')

    print("*"*50)
    print("Input shape:", input_.shape)
    print("Data type:", (input_.dtype))
    print("Output features size:", output[0].shape)
    print("*"*50)
    print('Average all batch time: %.3f ms'%(np.mean(all_timings)*1000))
    print('Average CPU batch time: %.3f ms'%(np.mean(cpu_timings)*1000))
    print('Average GPU batch time: %.3f ms'%(np.mean(gpu_timings)*1000))
    print("*"*50)