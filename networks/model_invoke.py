"""
Name : model_invoke.py
Author  : Hanat
Time    : 2019-12-20 14:20
Desc:
"""
import cv2
import numpy as np

import torch
import torch.nn as nn
from torchvision import models

# base_model = getattr(models, 'resnet50')
# base_model = base_model(pretrained=False)
# print("fesefaegfafvwr")

class NetWorkInvoker(nn.Module):

    def __init__(self, model_name='resnet50', embedding=512, pretrained=False):
        super(NetWorkInvoker, self).__init__()

        base_model = getattr(models, model_name)
        base_model = base_model(pretrained=pretrained)
        self.base_model = nn.Sequential(*list(base_model.children())[:-1])
        self._linear = nn.Linear(2048, embedding)

    def forward(self, x):
        x = self.base_model(x)
        x = torch.squeeze(x)
        y_pred = self._linear(x)
        return y_pred

    def load_weight(self, pretrained_path, devices):
        # if devices == torch.device('cpu'):
        self.base_model.load_state_dict(torch.load(pretrained_path, map_location=devices), strict=False)