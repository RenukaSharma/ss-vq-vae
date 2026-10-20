import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

from torch.autograd import Variable
from base.base_net import BaseNet
from torch.distributions.gamma import Gamma
from torch.distributions.normal import Normal


def sample_ggd(mu, beta):
    gamma = Gamma(1/beta, 1)
    y = gamma.rsample()
    p = torch.ones_like(mu)
    s = torch.bernoulli(p) - 0.5
    return mu + 2*s*(y**(1/beta))


class CIFAR10_LeNet_nano(BaseNet):

    def __init__(self, rep_dim=128):
        super().__init__()

        self.rep_dim = rep_dim

        self.pool = nn.MaxPool2d(2, 2)
        # for mu
        self.conv1 = nn.Conv2d(1, 32, 4, bias=False, stride=2, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)

        self.conv2 = nn.Conv2d(32, 64, 4, bias=False, stride=2, padding=1)
        nn.init.xavier_uniform_(self.conv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)

        self.conv3 = nn.Conv2d(64, 128, 4, bias=False, stride=2, padding=1)
        nn.init.xavier_uniform_(self.conv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)

        self.fc1 = nn.Linear(128 * 4 * 4, self.rep_dim, bias=False)
        nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('leaky_relu'))

        # self.fc2 = nn.Linear(128 * 4 * 4, self.rep_dim, bias=False)
        # nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('leaky_relu'))

        self.fc3 = nn.Linear(128 * 4 * 4, self.rep_dim, bias=False)
        nn.init.xavier_uniform_(self.fc3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        
        # self.bn1d1 = torch.nn.BatchNorm1d(128)
        # self.classifier_layer1 = nn.Linear(self.rep_dim, 16, bias=False)
        # self.bn1d2 = torch.nn.BatchNorm1d(16)
        # self.fc3 = nn.Linear(16, 1, bias=False)
        
        # self.classifier_layer2 = nn.Linear(16, 1, bias=False)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        x = x.view(-1, 1, 32, 32)
        x = self.conv1(x)
        x = F.leaky_relu(self.bn2d1(x))
        x = self.conv2(x)
        x = F.leaky_relu(self.bn2d2(x))
        x = self.conv3(x)
        x = F.leaky_relu(self.bn2d3(x))
        x = x.view(int(x.size(0)), -1)

        # x_mu = self.fc1(x)
        # x_alpha = 2*self.sigmoid(self.fc2(x)) - 1
        # x_beta = 2*self.sigmoid(self.fc3(x)) - 1

        x_mu = self.fc1(x)
        
        # x_alpha = self.tanh(self.fc2(x))
        x_beta = self.tanh(self.fc3(x))
        
        # return x_mu, x_alpha, x_beta  # x_encoded
        return x_mu, x_beta  # x_encoded

        

class CIFAR10_LeNet_Decoder_nano(BaseNet):

    def __init__(self, rep_dim=128):
        super().__init__()

        self.rep_dim = rep_dim

        self.d1 = nn.Linear(self.rep_dim, 128*2*4*4)
        nn.init.xavier_uniform_(self.d1.weight, gain=nn.init.calculate_gain('leaky_relu'))

        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd1 = nn.ReplicationPad2d(1)
        self.d2 = nn.Conv2d(128*2, 128, 3, 1)
        nn.init.xavier_uniform_(self.d2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn6 = nn.BatchNorm2d(128, 1.e-3)

        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd2 = nn.ReplicationPad2d(1)
        self.d3 = nn.Conv2d(128, 64, 3, 1)
        nn.init.xavier_uniform_(self.d3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn7 = nn.BatchNorm2d(64, 1.e-3)

        self.up3_mu = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd3_mu = nn.ReplicationPad2d(1)
        self.d4_mu = nn.Conv2d(64, 32, 3, 1)
        nn.init.xavier_uniform_(self.d4_mu.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn8_mu = nn.BatchNorm2d(32, 1.e-3)

        self.d5_mu = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.d5_mu.weight, gain=nn.init.calculate_gain('leaky_relu'))
        """
        self.up3_alpha = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd3_alpha = nn.ReplicationPad2d(1)
        self.d4_alpha = nn.Conv2d(64, 32, 3, 1)
        nn.init.xavier_uniform_(self.d4_alpha.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn8_alpha = nn.BatchNorm2d(32, 1.e-3)

        self.d5_alpha = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.d5_alpha.weight, gain=nn.init.calculate_gain('leaky_relu'))
        """
        self.up3_beta = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd3_beta = nn.ReplicationPad2d(1)
        self.d4_beta = nn.Conv2d(64, 32, 3, 1)
        nn.init.xavier_uniform_(self.d4_beta.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn8_beta = nn.BatchNorm2d(32, 1.e-3)

        self.d5_beta = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.d5_beta.weight, gain=nn.init.calculate_gain('leaky_relu'))

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        
    def forward(self, x):

        h1 = self.relu(self.d1(x))
        h1 = h1.view(-1,128*2 , 4, 4)
        h2 = self.leakyrelu(self.bn6(self.d2(self.pd1(self.up1(h1)))))
        h3 = self.leakyrelu(self.bn7(self.d3(self.pd2(self.up2(h2)))))
        h4_mu = self.leakyrelu(self.bn8_mu(self.d4_mu(self.pd3_mu(self.up3_mu(h3)))))

        # h4_alpha = self.leakyrelu(self.bn8_alpha(self.d4_alpha(self.pd3_alpha(self.up3_alpha(h3)))))

        h4_beta = self.leakyrelu(self.bn8_beta(self.d4_beta(self.pd3_beta(self.up3_beta(h3)))))
       
        x_mu = self.sigmoid(self.d5_mu(h4_mu))
        # x_alpha = self.tanh(self.d5_alpha(h4_alpha))
        x_beta = self.tanh(self.d5_beta(h4_beta))/1.6701

        # return x_mu, x_alpha, x_beta
        return x_mu, x_beta


class CIFAR10_LeNet_Autoencoder_nano(BaseNet):

    def __init__(self, rep_dim=128):
        super().__init__()

        self.rep_dim = rep_dim             
        
        self.encoder = CIFAR10_LeNet_nano(rep_dim=rep_dim)
        self.decoder = CIFAR10_LeNet_Decoder_nano(rep_dim=rep_dim)

        """
        Classifier results
        """   
        # self.bn1d1 = torch.nn.BatchNorm1d(128)
        # self.classifier_layer1 = nn.Linear(self.rep_dim, 16, bias=False)
        # self.bn1d2 = torch.nn.BatchNorm1d(16)
        # self.fc3 = nn.Linear(16, 1, bias=False)        
        # self.classifier_layer2 = nn.Linear(16, 1, bias=False)

    def forward(self, x, ablation_type='A', eps=1e-5):
        x_encoded_mu, x_encoded_beta = self.encoder(x)
        
        if ablation_type == 'A':
            sample = sample_ggd(x_encoded_mu, eps + torch.exp(x_encoded_beta))
            # sample = sample_ggd(x_encoded_mu, torch.ones_like(x_encoded_alpha), 1e-5 + torch.exp(x_encoded_beta))
        elif ablation_type == 'VAE':
            normal_sample = torch.randn(x_encoded_mu.shape).to('cuda' if torch.cuda.is_available() else 'cpu')
            sample = x_encoded_mu + torch.exp(x_encoded_alpha) * normal_sample
        else:
            sample = x_encoded_mu
        
        """
        Classifier results
        """        
        # class_out = self.bn1d1(sample)
        # class_out = self.classifier_layer1(class_out)
        # class_out = self.bn1d2(class_out)
        # class_out = self.classifier_layer2(class_out)
        # class_out = torch.sigmoid(class_out)

        x_recons_mu, x_recons_beta = self.decoder(sample)
        # x_recons_beta = torch.clamp(x_recons_beta, np.log(0.1), np.log(2.5))
        return x_encoded_mu, x_encoded_beta, x_recons_mu, x_recons_beta, sample
