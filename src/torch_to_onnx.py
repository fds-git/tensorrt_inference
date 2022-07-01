import torch
import onnxruntime
import numpy as np
from MobilenetV3PytorchModel import MobilenetV3
from ResnetPytorchModel import iresnet18, iresnet100
from MobileFaceNetPytorchModel import MobileFaceNet
import onnx
import argparse
from common_utils import image_shape_type


def save_to_onnx(model: object, destination_path: str, example_forward_input: torch.tensor, batch_type: str):
  '''Converting pytorch model to onnx
  Input parameters:
  model: object - pytorch model to convert
  destination_path: str - path to save onnx model
  example_forward_input: torch.tensor - tensor for tracing source model
  batch_type: str ('static' or 'dynamic') - result model will be work with static or dynamic batch size'''

  assert batch_type in ['dynamic', 'static']
  if batch_type == 'dynamic':
    dynamic_axes = {'input': {0: 'batch_size'},'output': {0: 'batch_size'}}
  else:
    dynamic_axes = None

  torch.onnx.export(model,
    example_forward_input,
    destination_path,
    export_params=True,
    opset_version=10,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes=dynamic_axes)

  # check the outputs of both models
  torch_out = model(example_forward_input)
  onnx_model = onnx.load(destination_path)
  onnx.checker.check_model(onnx_model)
  ort_session = onnxruntime.InferenceSession(destination_path, providers=['CPUExecutionProvider'])

  def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

  ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(example_forward_input)}
  ort_outs = ort_session.run(None, ort_inputs)
  np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)


if __name__ == "__main__":

  parser = argparse.ArgumentParser(description="To ONNX")
  parser.add_argument("-n", dest="name", help="onnx model name", required=False, default='model', type=str)
  parser.add_argument("-bt", dest="batch_type", help="dynamic or static (str)", required=True, choices=['static', 'dynamic'])
  parser.add_argument("-sh", dest="shape", help="example: 8, 3, 128, 128", required=True, type=image_shape_type)

  args = parser.parse_args()

  # load the model with weight to convert
  #model = MobilenetV3().eval().to('cpu')
  #model.load_state_dict(torch.load('../models/pytorch/pytorch_weights.pth'))
  #model.load_state_dict(torch.load('../models/pytorch/face_quality_cr-fiqa_mt_mae_3,6_ep_3.pth'))

  #model = iresnet100(num_features=512).eval().to('cpu')
  #model.load_state_dict(torch.load('../models/pytorch/glint360k_cosface_r100_fp16_0.1.pth'))

  # You need to load your own model and its weights
  model = iresnet18(num_features=512).eval().to('cpu')
  model.load_state_dict(torch.load('../models/pytorch/r18weights.pth'))

  #model = MobileFaceNet(512).eval().to('cpu')
  #model.load_state_dict(torch.load('../models/pytorch/MobileFace_Net'))

  example_forward_input = torch.randn(*args.shape, requires_grad=True).to('cpu')

  save_to_onnx(model=model, 
    destination_path=f'../models/onnx/{args.name}_{args.batch_type}_batch_{args.shape[0]}x{args.shape[1]}x{args.shape[2]}x{args.shape[3]}.onnx', 
    example_forward_input=example_forward_input,
    batch_type=args.batch_type)