"""
Name : test.py
Author  : BboyHanat
Time    : 2020/9/16 10:44 下午
Desc:
"""
import os
import cv2
import numpy as np
from thop import profile

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from networks.model_invoke import NetWorkInvoker



print("load ok")

class FastBaseTransform(torch.nn.Module):
    """
    Transform that does all operations on the GPU for super speed.
    This doesn't suppport a lot of config settings and should only be used for production.
    Maintain this as necessary.
    """

    def __init__(self, size):
        super(FastBaseTransform, self).__init__()
        self._input_size = size
        self._transforms = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def forward(self, img: torch.Tensor):
        """

        :param img:
        :return:
        """
        b, h, w, c = img.size()
        img = img.permute(0, 3, 1, 2).contiguous()

        if h >= w:
            scale = self._input_size / h
            img = F.interpolate(img, size=(self._input_size, int(scale*w)), mode='bilinear', align_corners=False)
            top_pad = 0
            bottom_pad = 0
            new_w = img.size()[3]
            left_pad = (self._input_size - new_w) // 2
            right_pad = self._input_size - new_w - left_pad

        else:
            scale = self._input_size / w
            img = F.interpolate(img, size=(int(h*scale), self._input_size), mode='bilinear', align_corners=False)
            new_h = img.size()[2]
            top_pad = (self._input_size - new_h) // 2
            bottom_pad = self._input_size - new_h - top_pad
            left_pad = 0
            right_pad = 0

        pad2d = (left_pad, right_pad, top_pad, bottom_pad)
        img = F.pad(img, pad2d)
        img = img / 255.0
        img = img.squeeze()
        if c == 3:
            img = self._transforms(img)
        else:
            img = img.unsqueeze(dim=0)

        return img


def test(image_path):
    """

    :param image_path:
    :return:
    """
    net = NetWorkInvoker(model_name='resnet50', embedding=512)
    state_dict = torch.load("weights/",
                            map_location=torch.device('cpu'))
    net.load_state_dict(state_dict, strict=True)

    net = net.eval()
    images = [os.path.join(image_path, img_file) for img_file in os.listdir(image_path)]
    for img in images:
        try:
            image = cv2.imread(img, 1)
            # cv2.imshow("aaa", image)
            # cv2.waitKey()
            image = np.expand_dims(image, 0)
            image = torch.from_numpy(image).float()
            fast_transform = FastBaseTransform(size=512)
            image = fast_transform(image)
            image = image.unsqueeze(dim=0)
            with torch.no_grad():
                output = net(image)
            print(torch.argmax(output, dim=1).data.cpu().numpy().squeeze(), img.split('/')[-1])

        except:
            print(img)

if __name__ == "__main__":
    test('/Users/aidaihanati/TezignProject/ImageCritic/data/critic_data/test_bad')
