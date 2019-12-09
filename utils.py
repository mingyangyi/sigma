import torch
import torch.nn as nn
import torch.nn.functional as F
from moodels import *


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
    index_tmp = torch.tensor(index_tmp, dtype=torch.bool)

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


def sigmanet(sigma):
    return 10 * sigma * resnet.ResNet_cifar10(dataset='cifar10', depth=8)
    # return Sigma_net(sigma)


