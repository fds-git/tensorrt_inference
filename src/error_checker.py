import tensorrt as trt
import torch
import onnxruntime
import numpy as np
from MobilenetV3PytorchModel import MobilenetV3
from ResnetPytorchModel import iresnet18
import onnx
import pycuda.autoinit
import argparse
from common_utils import image_shape_type
from TRTUtils import TrtInfer, TrtDynamicInfer

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Check models' convergence")
    parser.add_argument("-on", dest="onnx_name", help="onnx model name", required=True, type=str)
    parser.add_argument("-tn", dest="trt_name", help="trt model name", required=True, type=str)
    parser.add_argument("-sh", dest="shape", help="example: 8, 3, 128, 128", required=True, type=image_shape_type)
    args = parser.parse_args()

    # Pytorch
    model = MobilenetV3().eval().to('cpu')
    model.load_state_dict(torch.load('../models/pytorch/pytorch_weights.pth'))
    example_forward_input = torch.randn(*args.shape, requires_grad=True).to('cpu')
    torch_out = model(example_forward_input)

    # Onnx
    onnxmodel_path = '../models/onnx/' + args.onnx_name
    onnx_model = onnx.load(onnxmodel_path)
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession(onnxmodel_path, providers=['CPUExecutionProvider'])

    def to_numpy(tensor):
      return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(example_forward_input)}
    ort_outs = ort_session.run(None, ort_inputs)

    # TRT
    trt_infer = TrtInfer(path_to_model='../models/trt/' + args.trt_name)
    trt_output = trt_infer.inference(to_numpy(example_forward_input).astype(np.float32))

    # Checking
    print('Pytorch sv Onnx')
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print('Pytorch sv TRT')
    np.testing.assert_allclose(to_numpy(torch_out), trt_output[0].reshape((args.shape[0], -1)), rtol=1e-03, atol=1e-05)