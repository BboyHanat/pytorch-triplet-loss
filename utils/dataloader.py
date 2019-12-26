"""
Name : dataloader.py
Author  : Hanat
Contect : hanati@tezign.com
Time    : 2019-12-18 14:05
Desc:
"""
import os
import cv2
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class DataGenerate(object):
    def __init__(self, sample_sequence):
        self.sample_sequence = sample_sequence
        self.length = len(sample_sequence)
        self.seek = 0
        self.transform_train = transforms.Compose([transforms.RandomAffine((-10, 10), translate=(0.01, 0.01), scale=(0.9, 1.1), shear=(-5, 5)),
                                                   transforms.RandomHorizontalFlip(p=0.3),
                                                   transforms.RandomVerticalFlip(p=0.3)]
                                                  )

    def __call__(self, *args, **kwargs):
        return self._get_next().__next__()

    def _get_next(self):
        while True:
            if self.seek >= self.length:
                self.seek = 0
                random.shuffle(self.sample_sequence)
            file_path = self.sample_sequence[self.seek]
            img = cv2.imread(file_path)
            img = Image.fromarray(img)
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
        self.seek = 0

        self._transforms = transforms.Compose([transforms.Resize((init_h, init_w), interpolation=2),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
                                              )

    def __getitem__(self, index):
        """

        :param index:
        :return:
        """
        if (index + 1) % (self.class_num - 1) == 0 and (index + 1) >= (self.class_num - 1):
            random.shuffle(self.class_index)
            self.seek = 0

        seek1 = self.seek if self.seek < self.class_num else self.seek % self.class_num
        seek2 = self.seek + 1 if self.seek + 1 < self.class_num else (self.seek + 1)% self.class_num
        anchor_pos = self.class_index[seek1]
        neg = self.class_index[seek2]
        anchor_img = self.class_images_generator[anchor_pos]()
        pos_img = self.class_images_generator[anchor_pos]()
        neg_img = self.class_images_generator[neg]()
        anchor_img = self._transforms(anchor_img)
        pos_img = self._transforms(pos_img)
        neg_img = self._transforms(neg_img)
        self.seek += 2
        return torch.tensor([anchor_img, pos_img, neg_img])

    def __len__(self):
        return (self.class_num - 1) * self.augment_num

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
