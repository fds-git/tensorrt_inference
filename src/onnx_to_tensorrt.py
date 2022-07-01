import os
import tensorrt as trt
import sys
from typing import Tuple, Union
import logging
import argparse
from TRTUtils import build_engine_onnx, save_engine

# Based on code from NVES_R's response at
# https://forums.developer.nvidia.com/t/segmentation-fault-when-creating-the-trt-builder-in-python-works-fine-with-trtexec/111376


def image_shape_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    image_shape =  tuple(mapped_int)
    assert len(image_shape) == 3
    return image_shape


def convert_onnx(input_onnx: Union[str, bytes], engine_file_path: str, min_shape: tuple, opt_shape: tuple, max_shape: tuple, force_fp16: bool):
    '''
    Function for converting onnx-models, that work with images, to tensorrt format
    It may be used for working with onnx-models, that are able to process data with fixed number of channels, width and high  
    and with dynamic or static batch-size
    Input parameters:
    input_onnx: Union[str, bytes] - source model
    engine_file_path: str - path to save tensorrt object
    min_shape: tuple - min data shape (batch, channel, heigh, width)
    opt_shape: tuple - opt data shape (batch, channel, heigh, width)
    max_shape: tuple - max data shape (batch, channel, heigh, width)
    force_fp16: bool - use optimization to fp16
    '''

    input_onnx = f'../models/onnx/{args.source}'
    onnx_obj = None
    if isinstance(input_onnx, str):
        with open(input_onnx, "rb") as f:
            onnx_obj = f.read()
    elif isinstance(input_onnx, bytes):
        onnx_obj = input_onnx

    engine = build_engine_onnx(input_onnx=onnx_obj,
        force_fp16=force_fp16, 
        min_shape=min_shape, 
        opt_shape=opt_shape, 
        max_shape=max_shape)

    input_shape = engine.get_binding_shape(index=0)

    # if source model has dynamic batch size we need put information about min opt max batch sizes to file name for convenient
    # with other information
    if input_shape[0] == -1:
        min_opt_max_shapes = engine.get_profile_shape(profile_index=0, binding=0)
        min_batch = min_opt_max_shapes[0][0]
        opt_batch = min_opt_max_shapes[1][0]
        max_batch = min_opt_max_shapes[2][0]

        destination = engine_file_path + f'_{min_batch}-{opt_batch}-{max_batch}x{input_shape[1]}x{input_shape[2]}x{input_shape[3]}_{args.precision}.trt'
    else:
        destination = engine_file_path + f'_{input_shape[0]}x{input_shape[1]}x{input_shape[2]}x{input_shape[3]}_{args.precision}.trt'

    assert not isinstance(engine, type(None))
    with open(destination, "wb") as f:
        f.write(engine.serialize())



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="To TRT")
    parser.add_argument("-s", dest="source", help="onnx model name", required=True, type=str)
    parser.add_argument("-n", dest="name", help="trt model name", required=False, default='model', type=str)
    parser.add_argument("-prec", dest="precision", help="fp16 or fp32", required=True, choices=['fp16', 'fp32'])
    parser.add_argument("-min", dest="min_batch", required=False, type=int, default=None)
    parser.add_argument("-opt", dest="opt_batch", required=False, type=int, default=None)
    parser.add_argument("-max", dest="max_batch", required=False, type=int, default=None)
    parser.add_argument("-sh", dest="image_shape", help="example: 3, 128, 128", required=False, type=image_shape_type, default=None)

    args = parser.parse_args()
    if ((args.min_batch == None) or (args.opt_batch == None) or (args.max_batch == None) or (args.image_shape == None)):
        logging.warning('One of arguments (min, opt, max, sh) is missed. No optimization profile will be created. ' + 
                        ' Ensure that your onnx model has only static dimentions')
        min_shape = None
        opt_shape = None
        max_shape = None
    else:
        logging.warning(f'Will be created an optimization profile with min batch={args.min_batch}, opt batch={args.opt_batch}, ' + 
            f'max batch={args.max_batch}. Ensure that your onnx model\'s batch dimention is dynamic')

        min_shape = (args.min_batch,) + args.image_shape 
        opt_shape = (args.opt_batch,) + args.image_shape
        max_shape = (args.max_batch,) + args.image_shape

    force_fp16 = True if args.precision == 'fp16' else False

    destination = f'../models/trt/{args.name}'

    convert_onnx(args.source, destination, 
        force_fp16=force_fp16, 
        min_shape=min_shape, 
        opt_shape=opt_shape, 
        max_shape=max_shape)