"""
Name : test.py
Author  : BboyHanat
Time    : 2020/9/16 10:44 下午
Desc:
"""
import os
import cv2
import random
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader

from networks.model_invoke import NetWorkInvoker
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import seaborn as sns

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


def get_class_names(image_root):
    """

    :param image_root:
    :return:
    """
    class_index = [cls for cls in os.listdir(image_root) if os.path.isdir(os.path.join(image_root, cls))]
    class_index.sort()
    return class_index

def get_class_samples(path, format_list):
    class_samples = [os.path.join(path, cls) for cls in os.listdir(path) if os.path.isfile(os.path.join(path, cls)) and cls.split('.')[-1] in format_list]
    return class_samples


class DataGenerate(object):
    def __init__(self, sample_sequence, size=512):
        self.sample_sequence = sample_sequence
        self.length = len(sample_sequence)
        self.loc = 0
        self.transform_train = FastBaseTransform(size=size)

    def __call__(self, *args, **kwargs):
        return self._get_next().__next__()

    def _get_next(self):
        while True:
            if self.loc >= self.length:
                self.loc = 0
                random.shuffle(self.sample_sequence)
            file_path = self.sample_sequence[self.loc]
            img = cv2.imread(file_path)
            img = np.expand_dims(img, 0)
            img = torch.from_numpy(img).float()
            img = self.transform_train(img)
            yield img

    def __len__(self):
        return len(self.sample_sequence)

    def set_seek(self, loc):
        self.loc = loc


def test(image_path):
    """

    :param image_path:
    :return:
    """
    sns.set()
    class_names = get_class_names(image_path)
    class_images_generators = list()
    for class_path in class_names:
        path_to_image = os.path.join(image_path, class_path)
        class_samples_generator = DataGenerate(get_class_samples(path_to_image, format_list=['png', 'jpg']), size=512)
        class_images_generators.append(class_samples_generator)

    net = NetWorkInvoker(model_name='resnet50', embedding=512)
    state_dict = torch.load("weights/triplet/resnet50_feature_model_e0.pth", map_location=torch.device('cpu'))
    net.load_state_dict(state_dict, strict=True)
    device = torch.device('cuda:0')
    net = net.to(device)
    net = net.eval()
    tsne = TSNE(n_components=2, learning_rate=100)

    cls_outputs = list()
    for cls_index in range(len(class_images_generators)):
        for img_index in range(len(class_images_generators[cls_index])):
            img_tensor = class_images_generators[cls_index]()
            img_tensor = img_tensor.unsqueeze(dim=0)
            img_tensor = img_tensor.to(device)

            output = net(img_tensor)
            output = output.data.cpu().numpy().squeeze()
            cls_outputs.append(output)
    cls_outputs = np.asarray(cls_outputs).squeeze()
    output_2d = tsne.fit_transform(cls_outputs).squeeze()
    print(output_2d.shape)

    output_2d[:, 0] = output_2d[:, 0] + np.abs(output_2d[:, 0].min())
    output_2d[:, 0] = output_2d[:, 0] / output_2d[:, 0].max()
    output_2d[:, 1] = output_2d[:, 1] + np.abs(output_2d[:, 1].min())
    output_2d[:, 1] = output_2d[:, 1] / output_2d[:, 1].max()

    last_location = 0
    for cls_index in range(len(class_images_generators)):
        step = len(class_images_generators[cls_index])
        print(step, last_location)
        sns.scatterplot(x=output_2d[last_location:last_location+step, 0], y=output_2d[last_location:last_location+step, 1])
        last_location += step

    plt.savefig('1.png')



if __name__ == "__main__":
    test('/data/User/hanati/image_critic_dataset/triplet_visualize')
