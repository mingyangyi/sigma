import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
__all__ = ['mnist_lenet']

# class mnist_model(nn.Module):
#     def __init__(self):
#         super(mnist_model, self).__init__()
#         self.conv1 = nn.Conv2d(1, 20, 5, 1)
#         self.bn1 = nn.BatchNorm2d(20)
#         self.conv2 = nn.Conv2d(20, 50, 5, 1)
#         self.bn2 = nn.BatchNorm2d(50)
#         self.fc1 = nn.Linear(4 * 4 * 50, 500)
#         self.fc1_bn1 = nn.BatchNorm1d(500)
#         self.fc2 = nn.Linear(500, 10, bias=False)
#         #self.fc3 = nn.Linear(10, 10, bias=False)
#         #self.fc2_bn2 = nn.BatchNorm1d(10)
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = F.relu(out)
#         out = F.max_pool2d(out, 2)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = F.relu(out)
#         out = F.max_pool2d(out, 2)
#
#         out = out.view(-1, 4 * 4 * 50)
#         out = F.relu(self.fc1_bn1(self.fc1(out)))
#         out = out / out.norm(2, 1).unsqueeze(1)
#         out = self.fc2(out)
#         # out = self.fc3(out)
#         #out = nn.Softmax(out)
#
#         return out


class mnist_model(nn.Module):
    def __init__(self):
        super(mnist_model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # self.bn1 = nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(32, 32, 3, 1)
        self.conv3 = nn.Conv2d(32, 64, 3)
        # self.bn2 = nn.BatchNorm2d(50)
        self.conv4 = nn.Conv2d(64, 64, 3)
        # self.fc1_bn1 = nn.BatchNorm1d(500)
        self.fc1 = nn.Linear(64 * 4 * 4, 200)
        self.fc2 = nn.Linear(200, 200)
        self.drop1 = nn.Dropout()
        self.fc3 = nn.Linear(200, 10)
        #self.fc3 = nn.Linear(10, 10, bias=False)
        #self.fc2_bn2 = nn.BatchNorm1d(10)

    def forward(self, x):
        out = self.conv1(x)
        # out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = F.max_pool2d(out, 2)
        # out = self.bn2(out)
        out = self.conv3(out)
        out = F.relu(out)
        out = self.conv4(out)
        out = F.relu(out)
        out = F.max_pool2d(out, 2)

        out = out.view(-1, 4 * 4 * 64)
        # out = F.relu(self.fc1_bn1(self.fc1(out)))
        out = F.relu(self.fc1(out))
        # out = self.drop1(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        # out = out / out.norm(2, 1).unsqueeze(1)
        # out = self.fc3(out)
        #out = nn.Softmax(out)

        return out


def mnist_lenet(**kwargs):
    return mnist_model()