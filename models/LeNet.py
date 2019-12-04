import torch.nn as nn
import torch.nn.functional as F

__all__ = ['lenet_cifar']


# class LeNet(nn.Module):
#     def __init__(self):
#         super(LeNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5, bias=False)
#         self.bn1 = nn.BatchNorm2d(6)
#         self.conv2 = nn.Conv2d(6, 16, 5, bias=False)
#         self.bn2 = nn.BatchNorm2d(16)
#         self.fc1 = nn.Linear(16*5*5, 120, bias=False)
#         self.bn3 = nn.BatchNorm1d(120)
#         self.fc2 = nn.Linear(120, 84, bias=False)
#         self.bn4 = nn.BatchNorm1d(84)
#         self.fc3 = nn.Linear(84, 10, bias=False)
#         #self.bn5 = nn.BatchNorm1d(10)
#
#     def forward(self, x):
#         out = F.relu(self.conv1(x))
#         out = self.bn1(out)
#         out = F.max_pool2d(out, 2)
#         out = F.relu(self.conv2(out))
#         out = self.bn2(out)
#         out = F.max_pool2d(out, 2)
#         out = out.view(out.size(0), -1)
#         out = F.relu(self.fc1(out))
#         out = self.bn3(out)
#         out = F.relu(self.fc2(out))
#         out = self.bn4(out)
#         out = self.fc3(out)
#         #out = self.bn5(out)
#
#         return out

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=3, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.relu(out)

        return out


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feats(x)
        x = x.view(x.size(0), -1)
        x = x / x.norm(2, 1).unsqueeze(1)
        x = self.fc(x)

        return x


class LeNet_Cifar10(LeNet):

    def __init__(self, num_classes=10,
                 block=Bottleneck, depth=11):
        super(LeNet_Cifar10, self).__init__()
        self.inplanes = 64
        n = int((depth - 2) / 9)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        # self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = lambda x: x
        self.layer1 = self._make_layer(block, 128, n, stride=1)
        self.layer2 = self._make_layer(block, 256, n, stride=1)
        self.layer3 = self._make_layer(block, 512, n, stride=2)
        self.layer4 = lambda x: x
        self.avgpool = nn.AvgPool2d(8, stride=2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.feats = nn.Sequential(self.conv1,
                                   # self.bn1,
                                   self.relu,
                                   self.layer1,
                                   self.layer2,
                                   self.layer3,
                                   self.avgpool)


def lenet_cifar(**kwargs):
    num_classes, depth, dataset = map(
        kwargs.get, ['num_classes', 'depth', 'dataset'])

    num_classes = num_classes or 10
    depth = depth or 11
    return LeNet_Cifar10(num_classes=num_classes, block=Bottleneck, depth=depth)
