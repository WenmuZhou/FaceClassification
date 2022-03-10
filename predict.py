# -*- coding: utf-8 -*-
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Description: Evaluate the face classification models ('MiniXception' or 'SimpleCNN').
"""

import os
import logging
import paddle
import numpy as np
from models.simple_cnn import SimpleCNN
from models.mini_xception import MiniXception
import cv2
from argparse import ArgumentParser
from config.confg import parse_args
from data.dataset import preprocess_input


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def main():
    label_dict = {0: 'woman', 1: 'man'}
    # Initialize the model ('MiniXception' or 'SimpleCNN')
    if args.model_name == "MiniXception":
        model = MiniXception(n_classes=2, in_channels=1)
        img_size = (64,64)
        in_channels = 0
    else:
        model = SimpleCNN(n_classes=2, in_channels=3)
        img_size = (48,48)
        in_channels = 1

    # Load the model state dict
    model_state_dict = paddle.load(args.model_state_dict)
    model.set_state_dict(model_state_dict)

    model.eval()

    img = cv2.imread(args.img_path, in_channels)
    img = cv2.resize(img, img_size)
    img = preprocess_input(img)
    if len(img.shape)==2:
        img = np.expand_dims(img,axis=-1)
    img = np.transpose(img, (2, 1, 0))
    img = np.expand_dims(img,axis=0)
    tensor= paddle.to_tensor(img)
    pred = model(tensor)[0]
    scores = paddle.nn.functional.softmax(pred).numpy()
    label = scores.argmax()
    logging.info('label: {}, score: {}'.format(label_dict[label],scores[label]))

if __name__ == '__main__':
    parser = ArgumentParser("Eval parameters")
    parser.add_argument('--img_path', '-c', type=str, default='config/conf.yaml', help='Path to the config.')
    parser.add_argument('--model_name', '-m', type=str, choices=['MiniXception', 'SimpleCNN'], help='Choose a model.')
    parser.add_argument('--model_state_dict', '-msd', type=str, help='Path to the model parameters.')

    args = parser.parse_args()
    paddle.device.set_device('cpu')
    main()

