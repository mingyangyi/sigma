import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def create_set(trainset, sigma):
    set_return = [trainset[i] + (sigma[i], ) for i in range(len(trainset))]
    return set_return


def list_to_tensor(trainset):
    inputs = torch.stack([trainset[i][0] for i in range(len(trainset))], 0)
    targets = torch.stack([torch.tensor(trainset[i][1]) for i in range(len(trainset))], 0)
    sigma = torch.stack([trainset[i][2] for i in range(len(trainset))], 0)

    return [inputs, targets, sigma]


def gen_index(index, length):
    index_tmp = [0 for i in range(length)]
    for i in index:
        index_tmp[i] = 1
    index_tmp = torch.tensor(index_tmp, dtype=torch.uint8)

    return index_tmp


def cal_index(indices, subindices):
    num = 0
    indices_tmp = torch.zeros_like(indices)
    indices_tmp.data.copy_(indices)
    for i in range(len(indices)):
        if indices[i].item() == 1:
            if subindices[num].item() != 1:
                indices_tmp[i] = 0
            elif subindices[num].item() == 1:
                indices_tmp[i] = 1
            num += 1
        else:
            continue

    return indices_tmp


class Sigma_net(nn.Module):
    def __init__(self, sigma):
        super(Sigma_net, self).__init__()
        self.sigma = sigma
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.fc1 = nn.Linear(16 * 32 * 32, 512, bias=True)
        self.fc2 = nn.Linear(512, 512, bias=True)
        self.fc3 = nn.Linear(512, 512, bias=True)
        self.fc4 = nn.Linear(512, 1, bias=True)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.fc1(out.view(out.size()[0], -1)))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        out = torch.sigmoid(out) * self.sigma * 2

        return out


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def init_model(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, sigma):
        super(ResNet, self).__init__()
        self.sigma = sigma

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feats(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = 2 * self.sigma * F.softmax(x, dim=1).max(1)[0]

        return x


class ResNet_cifar10(ResNet):

    def __init__(self, sigma=0.25, num_classes=10,
                 block=BasicBlock, depth=20):
        super(ResNet_cifar10, self).__init__(sigma=sigma)
        self.inplanes = 16
        n = int((depth - 2) / 6)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = lambda x: x
        self.layer1 = self._make_layer(block, 16, n, stride=1)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.layer4 = lambda x: x
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.feats = nn.Sequential(self.conv1,
                                   self.bn1,
                                   self.relu,
                                   self.layer1,
                                   self.layer2,
                                   self.layer3,
                                   self.avgpool)
        init_model(self)


def sigmanet(sigma):
    return ResNet_cifar10(sigma=sigma, num_classes=2, block=BasicBlock, depth=20)
    # return Sigma_net(sigma)


