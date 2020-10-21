"""
Name : VGG16.py
Author  : BboyHanat
Time    : 2020/9/27 4:59 下午
Desc:
"""

import torch
import torch.nn as nn

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        shape = x.size()
        x = x.view((shape[0], shape[1], -1))
        x = x.sum(dim=2)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False, in_channels=3):
    layers = []
    in_channels = in_channels
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                if i == len(cfg) - 2:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.Tanh()]
                else:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def vgg13_bn(class_num=1000, in_channels=3):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        class_num (int): num_classes
        in_channels (int): in_channels
    """
    model = VGG(make_layers(cfgs['A'], batch_norm=True, in_channels=in_channels), num_classes=class_num)
    return model


if __name__ == '__main__':
    vgg13 = vgg13_bn(class_num=2)
    state_vgg = vgg13.state_dict()
    print(state_vgg.keys())

    order_key = ['features.0.weight', 'features.0.bias', 'features.1.weight', 'features.1.bias', 'features.1.running_mean', 'features.1.running_var', 'features.1.num_batches_tracked', 'features.3.weight', 'features.3.bias', 'features.4.weight', 'features.4.bias', 'features.4.running_mean', 'features.4.running_var', 'features.4.num_batches_tracked', 'features.7.weight', 'features.7.bias', 'features.8.weight', 'features.8.bias', 'features.8.running_mean', 'features.8.running_var', 'features.8.num_batches_tracked', 'features.10.weight', 'features.10.bias', 'features.11.weight', 'features.11.bias', 'features.11.running_mean', 'features.11.running_var', 'features.11.num_batches_tracked', 'features.14.weight', 'features.14.bias', 'features.15.weight', 'features.15.bias', 'features.15.running_mean', 'features.15.running_var', 'features.15.num_batches_tracked', 'features.17.weight', 'features.17.bias', 'features.18.weight', 'features.18.bias', 'features.18.running_mean', 'features.18.running_var', 'features.18.num_batches_tracked', 'features.20.weight', 'features.20.bias', 'features.21.weight', 'features.21.bias', 'features.21.running_mean', 'features.21.running_var', 'features.21.num_batches_tracked', 'features.24.weight', 'features.24.bias', 'features.25.weight', 'features.25.bias', 'features.25.running_mean', 'features.25.running_var', 'features.25.num_batches_tracked', 'features.27.weight', 'features.27.bias', 'features.28.weight', 'features.28.bias', 'features.28.running_mean', 'features.28.running_var', 'features.28.num_batches_tracked', 'features.30.weight', 'features.30.bias', 'features.31.weight', 'features.31.bias', 'features.31.running_mean', 'features.31.running_var', 'features.31.num_batches_tracked', 'features.34.weight', 'features.34.bias', 'features.35.weight', 'features.35.bias', 'features.35.running_mean', 'features.35.running_var', 'features.35.num_batches_tracked', 'features.37.weight', 'features.37.bias', 'features.38.weight', 'features.38.bias', 'features.38.running_mean', 'features.38.running_var', 'features.38.num_batches_tracked', 'features.40.weight', 'features.40.bias', 'features.41.weight', 'features.41.bias', 'features.41.running_mean', 'features.41.running_var', 'features.41.num_batches_tracked']

    state_dict = torch.load("/Users/aidaihanati/TezignProject/ImageCritic/weights/vgg16_bn.pth",
                            map_location=torch.device('cpu'))

    new_state_dict = {key: state_dict[key] for key in state_dict.keys() if key in order_key}

    torch.save(new_state_dict, "vgg16_bn_without_classifier.pth")

    vgg13.load_state_dict(new_state_dict, strict=False)



    print('load ok')

    import numpy as np
    input = np.zeros((10, 3, 224, 224), dtype=np.float32)
    input = torch.from_numpy(input)

    output = vgg13(input)
    print(output.size())