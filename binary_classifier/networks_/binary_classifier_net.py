import torch
import torch.nn as nn

from .cifar10_LeNet_vae import CIFAR10_LeNet
from .mnist_LeNet_vae import MNIST_LeNet
from .fmnist_LeNet_vae import FashionMNIST_LeNet
from base.base_net import BaseNet

class BinaryClassifierNet(BaseNet):

    def __init__(self, net_name):
        super().__init__()

        if net_name == 'mnist_classifier':
            self.network = MNIST_LeNet()
        if net_name == 'fmnist_classifier':
            self.network = FashionMNIST_LeNet()
        if net_name == 'cifar10_classifier':
            self.network = CIFAR10_LeNet()

        self.linear = nn.Linear(self.network.rep_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x, _, _  = self.network(x)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x