"""
Name : dataloader.py
Author  : Hanat
Time    : 2019-12-18 14:05
Desc:
"""
import os
import cv2
import random
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms


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


class DataGenerate(object):
    def __init__(self, sample_sequence):
        self.sample_sequence = sample_sequence
        self.length = len(sample_sequence)
        self.seek = 0
        self.transform_train = FastBaseTransform(size=512)

    def __call__(self, *args, **kwargs):
        return self._get_next().__next__()

    def _get_next(self):
        while True:
            if self.seek >= self.length:
                self.seek = 0
                random.shuffle(self.sample_sequence)
            file_path = self.sample_sequence[self.seek]
            img = cv2.imread(file_path)
            img = np.expand_dims(img, 0)
            img = torch.from_numpy(img).float()
            img = self.transform_train(img)
            yield img

    def set_seek(self, seek):
        self.seek = seek


class TripletDataset(Dataset):

    def __init__(self, image_root, format_list, augment_num=1000, init_h=512, init_w=512):
        """

        :param txt: train txt file path
        :param image_root: image file path
        :param init_h: 32
        :param init_w: 448
        """
        if format_list is None:
            format_list = ['jpg']

        class_names = self.get_class_names(image_root)
        self.class_images_generator = list()
        for class_path in class_names:
            path_to_image = os.path.join(image_root, class_path)
            class_samples_generator = DataGenerate(self.get_class_samples(path_to_image, format_list))
            self.class_images_generator.append(class_samples_generator)

        self.class_num = len(class_names)
        self.class_index = list(np.arange(self.class_num))
        self.augment_num = augment_num

        self._transforms = transforms.Compose([transforms.Resize((init_h, init_w), interpolation=2),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
                                              )

    def __getitem__(self, index):
        """

        :param index:
        :return:
        """
        loc1 = index if index < self.class_num else index % self.class_num     # seek of pos image
        loc2 = index + 1 if index + 1 < self.class_num else (index + 1) % self.class_num    # seek of pos image
        anchor_pos = self.class_index[loc1]
        neg = self.class_index[loc2]
        anchor_img = self.class_images_generator[anchor_pos]()
        pos_img = self.class_images_generator[anchor_pos]()
        neg_img = self.class_images_generator[neg]()
        random.shuffle(self.class_index)
        return anchor_img, pos_img, neg_img

    def __len__(self):
        return self.class_num

    @staticmethod
    def get_class_names(image_root):
        """

        :param image_root:
        :return:
        """
        class_index = [cls for cls in os.listdir(image_root) if os.path.isdir(os.path.join(image_root, cls))]
        class_index.sort()
        return class_index

    @staticmethod
    def get_class_samples(path, format_list):
        class_samples = [os.path.join(path, cls) for cls in os.listdir(path) if os.path.isfile(os.path.join(path, cls)) and cls.split('.')[-1] in format_list]
        return class_samples


if __name__ == '__main__':
    image_root = '../datasets'
    triplet = TripletDataset(image_root, format_list=['png'])

    for index, t in enumerate(triplet):
        print(index, t[0].shape, t[1].shape, t[2].shape)
