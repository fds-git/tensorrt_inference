import numpy as np
import torch
import time
import numpy as np
import torch.backends.cudnn as cudnn
import onnxruntime
import argparse
from common_utils import image_shape_type

def benchmark(session, input_shape=(1, 3, 128, 128), dtype='fp32', nruns=10000):

    timings = []

    # onnx model with trtexecution provider needs only fp32 data even though in optimized with fp16
    input_data = np.random.rand(*input_shape)
    dtype = np.float16 if dtype == 'fp16' else np.float32
    input_data = input_data.astype(dtype)

    for i in range(1, nruns+1):
        start_time = time.time()
        features = session.run(None, {session.get_inputs()[0].name:input_data})
        #torch.cuda.synchronize()
        end_time = time.time()
        timings.append(end_time - start_time)
        if i%1000==0:
            print('Iteration %d/%d, ave batch time %.2f ms'%(i, nruns, np.mean(timings)*1000))

    print("Input shape:", input_data.shape)
    print("Output features size:", features[0].shape)
    print('Average batch time: %.2f ms'%(np.mean(timings)*1000))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test ONNX")
    parser.add_argument("-n", dest="name", help="onnx model name", required=True, default='model', type=str)
    parser.add_argument("-p", dest="provider", help="cpu or cuda or trt", required=True, choices=['cpu', 'cuda', 'trt'])
    parser.add_argument("-sh", dest="shape", help="example: 8, 3, 128, 128", required=True, type=image_shape_type)
    parser.add_argument("-prec", dest="precision", help="fp16 or fp32 - only if provider = trt", required=True, choices=['fp16', 'fp32'])

    args = parser.parse_args()

    cudnn.benchmark = True
    print(onnxruntime.get_device())
    print(onnxruntime.__version__)

    path_to_model = f'../models/onnx/{args.name}'

    if args.provider == 'cpu':
        providers = ['CPUExecutionProvider']

    if args.provider == 'cuda':
        providers = [
        ('CUDAExecutionProvider', {
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
            'cudnn_conv_algo_search': 'DEFAULT',
            #'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True,
        })]

    if args.provider == 'trt':
        trt_fp16_enable = True if args.precision == 'fp16' else False
        print(trt_fp16_enable)
        providers = [
        ('TensorrtExecutionProvider', {
            'device_id': 0,
            'trt_max_workspace_size': 2147483648,
            'trt_fp16_enable': trt_fp16_enable,
        })]

    sess_opt = onnxruntime.SessionOptions()
    session = onnxruntime.InferenceSession(path_to_model, sess_options=sess_opt, providers=providers)

    benchmark(session, input_shape=shape, nruns=10000)