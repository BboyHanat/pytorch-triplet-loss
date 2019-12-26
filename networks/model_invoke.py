"""
Name : model_invoke.py
Author  : Hanat
Contect : hanati@tezign.com
Time    : 2019-12-20 14:20
Desc:
"""
import cv2
import numpy as np
import torch.nn as nn
from torchvision import models


class NetWorkInvoker(nn.Module):

    def __init__(self, model_name='resnet50', embedding=512, pretrained=True):

        self.base_model = getattr(models, model_name)
        self.base_model = nn.Sequential(*list(self.base_model(pretrained=pretrained))[:-1])
        self._linear = nn.Linear(2048, embedding)

    def forward(self, x):
        x = self.base_model(x)
        y_pred = self._linear(x)

        return y_pred
