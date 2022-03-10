# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
import cv2
import time
import json
import pickle
import argparse
import numpy as np
from data.dataset import preprocess_input

sys.path.insert(0, os.path.abspath('.'))

def str2bool(v):
    return v.lower() in ("True","true", "t", "1")

def parse_args():
    parser = argparse.ArgumentParser(description='Paddle Face Predictor')

    parser.add_argument(
        "--inference_model",
        type=str,
        required=False,
        help="paddle save inference model filename")
    parser.add_argument("--image_path", type=str, help="path to test image")
    parser.add_argument('--model_name', '-m', type=str, choices=['MiniXception', 'SimpleCNN'], help='Choose a model.')
    parser.add_argument("--benchmark", type=str2bool, default=False, help="Is benchmark mode")
    # params for paddle inferece engine
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--ir_optim", type=str2bool, default=True)
    parser.add_argument("--use_tensorrt", type=str2bool, default=False)
    parser.add_argument("--min_subgraph_size", type=int, default=15)
    parser.add_argument("--max_batch_size", type=int, default=1)
    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--gpu_mem", type=int, default=500)
    parser.add_argument("--enable_mkldnn", type=str2bool, default=False)
    parser.add_argument("--cpu_threads", type=int, default=10)
    args = parser.parse_args()
    return args

def get_infer_gpuid():
    cmd = "nvidia-smi"
    res = os.popen(cmd).readlines()
    if len(res) == 0:
        return None
    cmd = "env | grep CUDA_VISIBLE_DEVICES"
    env_cuda = os.popen(cmd).readlines()
    if len(env_cuda) == 0:
        return 0
    else:
        gpu_id = env_cuda[0].strip().split("=")[1]
        return int(gpu_id[0])


def init_paddle_inference_config(args):
    import paddle.inference as paddle_infer
    config = paddle_infer.Config(os.path.join(args.inference_model, 'inference.pdmodel'), os.path.join(args.inference_model, 'inference.pdiparams'))
    if hasattr(args, 'precision'):
        if args.precision == "fp16" and args.use_tensorrt:
            precision = paddle_infer.PrecisionType.Half
        elif args.precision == "int8":
            precision = paddle_infer.PrecisionType.Int8
        else:
            precision = paddle_infer.PrecisionType.Float32
    else:
        precision = paddle_infer.PrecisionType.Float32

    if args.use_gpu:
        gpu_id = get_infer_gpuid()
        if gpu_id is None:
            raise ValueError(
                "Not found GPU in current device. Please check your device or set args.use_gpu as False"
            )
        config.enable_use_gpu(args.gpu_mem, 0)
        if args.use_tensorrt:
            config.enable_tensorrt_engine(
                precision_mode=precision,
                max_batch_size=args.max_batch_size,
                min_subgraph_size=args.min_subgraph_size)
            # skip the minmum trt subgraph
            min_input_shape = {"x": [1, 3, 10, 10]}
            max_input_shape = {"x": [1, 3, 1000, 1000]}
            opt_input_shape = {"x": [1, 3, 112, 112]}
            config.set_trt_dynamic_shape_info(min_input_shape, max_input_shape,
                                            opt_input_shape)

    else:
        config.disable_gpu()
        cpu_threads = args.cpu_threads if  hasattr(args, "cpu_threads") else 10
        config.set_cpu_math_library_num_threads(cpu_threads)
        if args.enable_mkldnn:
            # cache 10 different shapes for mkldnn to avoid memory leak
            config.enable_mkldnn()
            config.set_mkldnn_cache_capacity(10)
            if args.precision == "fp16":
                config.enable_mkldnn_bfloat16()
    return config

def softmax(logits):
	e_x = np.exp(logits)
	probs = e_x / np.sum(e_x, axis=-1, keepdims=True)
	return probs

def paddle_inference(args):
    import paddle.inference as paddle_infer
    config =  init_paddle_inference_config(args)
    predictor = paddle_infer.create_predictor(config)

    input_names = predictor.get_input_names()
    input_handle = predictor.get_input_handle(input_names[0])
    
    label_dict = {0: 'woman', 1: 'man'}
    if args.model_name == "MiniXception":
        img_size = (64,64)
        in_channels = 0
    else:
        img_size = (48,48)
        in_channels = 1

    if args.benchmark:
        import auto_log
        pid = os.getpid()
        autolog = auto_log.AutoLogger(
            model_name="det",
            model_precision='fp32',
            batch_size=1,
            data_shape="dynamic",
            save_path="./output/auto_log.log",
            inference_config=config,
            pids=pid,
            process_name=None,
            gpu_ids=0,
            time_keys=[
                'preprocess_time', 'inference_time','postprocess_time'
            ],
            warmup=0)
        img = np.random.uniform(0, 255, [1, 3, img_size[0],img_size[1]]).astype(np.float32)
        input_handle.copy_from_cpu(img)
        for i in range(2):
            predictor.run()

    img = cv2.imread(args.image_path, in_channels)
    st = time.time()
    if args.benchmark:
        autolog.times.start()
    img = cv2.resize(img, img_size)
    img = preprocess_input(img)
    if len(img.shape)==2:
        img = np.expand_dims(img,axis=-1)
    img = np.transpose(img, (2, 1, 0))
    img = np.expand_dims(img,axis=0)

    if args.benchmark:
        autolog.times.stamp()

    input_handle.copy_from_cpu(img)

    predictor.run()

    output_names = predictor.get_output_names()
    output_handle = predictor.get_output_handle(output_names[0])
    output_data = output_handle.copy_to_cpu()[0]
    scores = softmax(output_data)
    label = scores.argmax()
    if args.benchmark:
        autolog.times.stamp()
        autolog.times.end(stamp=True)
        print('{}\t{}'.format(img_path,json.dumps(output_data)))
    print('paddle inference result: label: {}, score: {}'.format(label_dict[label],scores[label]))
    if args.benchmark:
        autolog.report()

if __name__ == '__main__':

    args = parse_args()

    assert os.path.exists(args.inference_model)
    paddle_inference(args)