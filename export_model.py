# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os
import sys

from argparse import ArgumentParser
import paddle
from paddle.jit import to_static

from models.simple_cnn import SimpleCNN
from models.mini_xception import MiniXception

def main(args):
    # build post process
    if args.model_name == "MiniXception":
        model = MiniXception(n_classes=2, in_channels=1)
        infer_shape = [1,64,64]
    else:
        model = SimpleCNN(n_classes=2, in_channels=3)
        infer_shape = [3,48,48]

    model_state_dict = paddle.load(args.model_state_dict)
    model.set_state_dict(model_state_dict)
    model.eval()

    model = to_static(
        model,
        input_spec=[
            paddle.static.InputSpec(
                shape=[None] + infer_shape, dtype="float32")
        ])
    os.makedirs(args.save_path,exist_ok=True)
    paddle.jit.save(model, os.path.join(args.save_path, "inference"))
    print("inference model is saved to {}".format(os.path.join(args.save_path, "inference")))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model_name', '-m', type=str, choices=['MiniXception', 'SimpleCNN'], help='Choose a model.')
    parser.add_argument('--model_state_dict', '-msd', type=str, help='Path to the model parameters.')
    parser.add_argument('--save_path', default="inference_models", help="inference model save path")
    args = parser.parse_args()
    main(args)
