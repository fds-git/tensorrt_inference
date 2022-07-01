import numpy as np
import torch
import time
import torchvision
import numpy as np
import torch.backends.cudnn as cudnn
from MobileFaceNetPytorchModel import MobileFaceNet
import argparse
from common_utils import image_shape_type


def benchmark(model, provider:str, input_shape=(1, 3, 128, 128), dtype='fp32', nwarmup=50, nruns=10000):
    input_data = torch.randn(input_shape)
    input_data = input_data.to(provider)
    model = model.eval().to(provider)
    if dtype=='fp16':
        input_data = input_data.half()
        
    print("Warm up ...")
    with torch.no_grad():
        for _ in range(nwarmup):
            features = model(input_data)
    torch.cuda.synchronize()
    print("Start timing ...")
    timings = []
    with torch.no_grad():
        for i in range(1, nruns+1):
            start_time = time.time()
            features = model(input_data)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
            if i%1000==0:
                print('Iteration %d/%d, ave batch time %.2f ms'%(i, nruns, np.mean(timings)*1000))


    print("Input shape:", input_data.size())
    print("Output features size:", features.size())
    print('Average batch time: %.2f ms'%(np.mean(timings)*1000))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test PyTorch")
    parser.add_argument("-p", dest="provider", help="cpu or cuda", required=True, choices=['cpu', 'cuda'])
    parser.add_argument("-prec", dest="precision", help="fp16 or fp32", required=True, choices=['fp16', 'fp32'])
    parser.add_argument("-sh", dest="shape", help="example: 8, 3, 128, 128", required=True, type=image_shape_type)

    args = parser.parse_args()

    cudnn.benchmark = True
    path_to_model = '../models/pytorch/MobileFace_Net'
    face_quality = MobileFaceNet(embedding_size=512)
    face_quality.load_state_dict(torch.load(path_to_model))

    benchmark(face_quality, input_shape=args.shape, dtype=args.precision, nruns=10000, provider=args.provider)