import torch
import torch.nn as nn
import torch.nn.functional as F


def create_set(trainset, sigma):
    set_return = [trainset[i] + (sigma[i], ) for i in range(len(trainset))]
    return set_return


def cal_index(indices, subindices):
    num = 0
    indices_tmp = torch.zeros_like(indices)
    indices_tmp.data.copy_(indices)
    for i in range(len(indices)):
        if indices[i] == 1:
            if subindices[num] != 1:
                indices_tmp[i] = 0
            elif subindices[num] != 1:
                indices_tmp[i] = 1
            num += 1
        else:
            continue

        return indices_tmp


class Sigma_net(nn.Module):
    def __init__(self):
        super(Sigma_net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.fc1 = nn.Linear(16 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.softmax(out, dim=1)

        return out.max(1)


def sigma_net():
    return Sigma_net()


