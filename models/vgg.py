'''
Modified from https://github.com/pytorch/vision.git
'''
import math

import torch.nn as nn
#import torch.nn.init as init

__all__ = ['vgg']#, 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    #'vgg19_bn', 'vgg19',
#]


class VGG_cifar10(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features):
        super(VGG_cifar10, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            #nn.Dropout(),
            nn.Linear(512, 512, bias=False),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(512, 512, bias=False),
            nn.ReLU(True),
            nn.Linear(512, 10, bias=False),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class VGG_cifar100(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features):
        super(VGG_cifar100, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            #nn.Dropout(),
            nn.Linear(512, 512, bias=False),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(512, 512, bias=False),
            nn.ReLU(True),
            nn.Linear(512, 100, bias=False),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v, affine=False), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}


def vgg(**kwargs):
    input_size, dataset, depth = map(
        kwargs.get, ['input_size', 'dataset', 'depth'])
    # if model == 'vgg11':
    #     """VGG 11-layer model (configuration "A")"""
    #     return VGG(make_layers(cfg['A']))

    if depth == 11 and dataset == 'cifar10':
        """VGG 11-layer model (configuration "A") with batch normalization"""
        return VGG_cifar10(make_layers(cfg['A'], batch_norm=True))
    if depth == 11 and dataset == 'cifar100':
        return VGG_cifar100(make_layers(cfg['A'], batch_norm=True))
    # if model == 'vgg13':
    #     """VGG 13-layer model (configuration "B")"""
    #     return VGG(make_layers(cfg['B']))

    if depth == 13 and dataset == 'cifar10':
        """VGG 13-layer model (configuration "A") with batch normalization"""
        return VGG_cifar10(make_layers(cfg['B'], batch_norm=True))
    if depth == 13 and dataset == 'cifar100':
        return VGG_cifar100(make_layers(cfg['B'], batch_norm=True))

    # if model == 'vgg16':
    #     """VGG 16-layer model (configuration "D")"""
    #     return VGG(make_layers(cfg['D']))

    if depth == 16 and dataset == 'cifar10':
        """VGG 16-layer model (configuration "A") with batch normalization"""
        return VGG_cifar10(make_layers(cfg['D'], batch_norm=True))
    if depth == 16 and dataset == 'cifar100':
        return VGG_cifar100(make_layers(cfg['D'], batch_norm=True))
    # if model == 'vgg19':
    #     """VGG 19-layer model (configuration "E")"""
    #     return VGG(make_layers(cfg['E']))

    if depth == 19 and dataset == 'cifar10':
        """VGG 19-layer model (configuration "E") with batch normalization"""
        return VGG_cifar10(make_layers(cfg['E'], batch_norm=True))
    if depth == 19 and dataset == 'cifar100':
        return VGG_cifar100(make_layers(cfg['E'], batch_norm=True))