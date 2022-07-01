import numpy as np
from TRTUtils import TrtDynamicInfer
import tensorrt as trt 
import time
import argparse
from common_utils import image_shape_type
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test TRT")
    parser.add_argument("-n", dest="name", help="model name", required=True, type=str)
    args = parser.parse_args()

    path_to_model = f'../models/trt/{args.name}'
    trt_infer = TrtDynamicInfer(path_to_model=path_to_model)

    random_shape = trt_infer.engine.get_binding_shape(index=0) # for model with dynamic batch is (-1, C, H, W)
    min_opt_max_shapes = trt_infer.engine.get_profile_shape(profile_index=0, binding=0)
    min_shape = min_opt_max_shapes[0]
    opt_shape = min_opt_max_shapes[1]
    max_shape = min_opt_max_shapes[2]

    input_shapes = [min_shape, opt_shape, max_shape, random_shape]
    
    print(" ")
    print(f'Experiments with next data shapes will be executed: {input_shapes}')
    print(f'Data shape with -1 batch dimention means that generated data will have random batch size from {min_shape[0]} to {max_shape[0]}')
    print(" ")

    for input_shape in input_shapes:
        cpu_timings, gpu_timings, all_timings = [], [], []
        print("*"*25, end=" ")
        print(f"Data shape = {input_shape}", end=" ")
        print("*"*25)
        nruns = 10000

        for i in range(1, nruns+1):
            if input_shape[0] == -1:
                random_input_shape = [*input_shape]
                random_input_shape[0] = random.randint(min_shape[0], max_shape[0])
                input_ = np.random.rand(*random_input_shape)
            else:
                input_ = np.random.rand(*input_shape)

            # In inference time you should put fp32 data into fp16 optimized model
            # othervise data proccesing time will be increased
            input_ = input_.astype(np.float32)
            output, start_time, start_gpu_time, end_gpu_time = trt_infer.inference_with_time(input_)

            cpu_timings.append(start_gpu_time - start_time)
            gpu_timings.append(end_gpu_time - start_gpu_time)
            all_timings.append(end_gpu_time - start_time)

            if i%1000==0:
                print(f'Iteration {i}/{nruns}, ave batch time: CPU: {np.mean(cpu_timings)*1000:.3}, GPU: {np.mean(gpu_timings)*1000:.3}, ALL: {np.mean(all_timings)*1000:.3}')

        if input_shape[0] == -1:
            print(" ")
            print("Input shape:", input_shape)
            print("Data type:", (input_.dtype))
            print("Output features size: random")
            print(" ")
        else:
            print(" ")
            print("Input shape:", input_.shape)
            print("Data type:", (input_.dtype))
            print("Output features size:", output[0].shape)
            print(" ")

        print('Average all batch time: %.3f ms'%(np.mean(all_timings)*1000))
        print('Average CPU batch time: %.3f ms'%(np.mean(cpu_timings)*1000))
        print('Average GPU batch time: %.3f ms'%(np.mean(gpu_timings)*1000))
        print(" ")