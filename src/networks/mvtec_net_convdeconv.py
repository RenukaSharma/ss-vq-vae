import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet
import numpy as np

from torch.autograd import Variable
from base.base_net import BaseNet
from torch.distributions.gamma import Gamma
from torch.distributions.normal import Normal

import sys

def sample_ggd(mu, alpha, beta):
    gamma = Gamma(1/beta, 1)
    y = gamma.rsample()
    p = torch.ones_like(mu)
    s = torch.bernoulli(p) - 0.5
    return mu + 2*alpha*s*(y**(1/beta))

class MVTec_net(BaseNet):

    def __init__(self, rep_dim=512):
        super().__init__()

        self.rep_dim = rep_dim
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 32, 5, bias=False, padding=0,stride=2)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=0)
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(64, 128, 5, bias=False, padding=0)
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.conv4 = nn.Conv2d(128, 256, 3, bias=False, padding=0)
        self.bn2d4 = nn.BatchNorm2d(256, eps=1e-04, affine=False)

        # self.conv5 = nn.Conv2d(256, 512, 3, bias=False, padding=0)
        # self.bn2d5 = nn.BatchNorm2d(512, eps=1e-04, affine=False)

        self.fc1 = nn.Linear(256 * 5 * 5, self.rep_dim, bias=False)
        self.fc2 = nn.Linear(256 * 5 * 5, self.rep_dim, bias=False)
        self.fc3 = nn.Linear(256 * 5 * 5, self.rep_dim, bias=False)
        # self.conv6= nn.Conv2d(512, self.rep_dim,1, bias=False)

        # self.conv6_alpha= nn.Conv2d(512, self.rep_dim,1, bias=False)

        # self.conv6_beta= nn.Conv2d(512, self.rep_dim,1, bias=False)

        # self.bn2d5_1 = nn.BatchNorm2d(self.rep_dim, eps=1e-04, affine=False)

    def forward(self, x):
        print("Init encoder:", x.shape)
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        print("Shape after conv 1:", x.shape)
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        print("Shape after conv 2:", x.shape)
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        print("Shape after conv 3:", x.shape)
        x = self.conv4(x)
        x = self.pool(F.leaky_relu(self.bn2d4(x)))
        print("Shape after conv 4:", x.shape)
        # x = self.conv5(x)
        # x = self.pool(F.leaky_relu(self.bn2d5(x)))
        # print("Shape after conv 5:", x.shape)
        # x_mu = self.conv6(x)
        # print("Shape after conv 6:", x.shape)
        x = x.view(int(x.size(0)), -1)
        x_mu = torch.sigmoid(self.fc1(x))
        x_alpha = torch.tanh(self.fc2(x))
        x_beta = torch.tanh(self.fc3(x))/1.6701
        return x_mu, x_alpha, x_beta

class MVTec_DecoderNet(BaseNet):

    def __init__(self, rep_dim=512):
        super().__init__()

        self.rep_dim = rep_dim
        self.pool = nn.MaxPool2d(2, 2)

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(int(self.rep_dim / (5 * 5)), 512, 3, bias=False, padding=0)
        # self.deconv1 = nn.ConvTranspose2d(self.rep_dim , 512, 3, bias=False, padding=1) 
        nn.init.xavier_uniform_(self.deconv1.weight, gain=nn.init.calculate_gain('leaky_relu'))        
        self.bn2d6 = nn.BatchNorm2d(512, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(512, 256, 3, bias=False, padding=0)
        nn.init.xavier_uniform_(self.deconv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d7 = nn.BatchNorm2d(256, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(256, 128, 3, bias=False, padding=1)
        nn.init.xavier_uniform_(self.deconv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d8 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        
        #mu
        self.deconv4_mu = nn.ConvTranspose2d(128, 64, 3, bias=False, padding=1)
        nn.init.xavier_uniform_(self.deconv4_mu.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d9_mu = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.deconv5_mu = nn.ConvTranspose2d(64, 32, 3, bias=False, padding=1)
        nn.init.xavier_uniform_(self.deconv5_mu.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d10_mu = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.deconv6_mu = nn.ConvTranspose2d(32, 3, 3, bias=False, padding=1)
        nn.init.xavier_uniform_(self.deconv6_mu.weight, gain=nn.init.calculate_gain('leaky_relu'))
        
        #alpha
        self.deconv4_alpha = nn.ConvTranspose2d(128, 64, 3, bias=False, padding=1)
        nn.init.xavier_uniform_(self.deconv4_alpha.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d9_alpha = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.deconv5_alpha = nn.ConvTranspose2d(64, 32, 3, bias=False, padding=1)
        nn.init.xavier_uniform_(self.deconv5_alpha.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d10_alpha = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.deconv6_alpha = nn.ConvTranspose2d(32, 3, 3, bias=False, padding=1)
        nn.init.xavier_uniform_(self.deconv6_alpha.weight, gain=nn.init.calculate_gain('leaky_relu'))
        

        #beta    
        self.deconv4_beta = nn.ConvTranspose2d(128, 64, 3, bias=False, padding=1)
        nn.init.xavier_uniform_(self.deconv4_beta.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d9_beta = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.deconv5_beta = nn.ConvTranspose2d(64, 32, 3, bias=False, padding=1)
        nn.init.xavier_uniform_(self.deconv5_beta.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d10_beta = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.deconv6_beta = nn.ConvTranspose2d(32, 3, 3, bias=False, padding=1)
        nn.init.xavier_uniform_(self.deconv6_beta.weight, gain=nn.init.calculate_gain('leaky_relu'))
        

    def forward(self, x):
        print("Init Decoder:", x.shape)
        # sys.exit()
        x = x.view(int(x.size(0)), int(self.rep_dim / (5 * 5)), 5, 5)
        x = F.leaky_relu(x)
        print("Init Decoder:", x.shape)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn2d6(x)), scale_factor=2)
        print("Shape after deconv 1:", x.shape)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn2d7(x)), scale_factor=2)
        print("Shape after deconv 2:", x.shape)
        x = self.deconv3(x)
        x = F.interpolate(F.leaky_relu(self.bn2d8(x)), scale_factor=2)
        print("Shape after deconv 3:", x.shape)

        x_mu = self.deconv4_mu(x)
        x_mu = F.interpolate(F.leaky_relu(self.bn2d9_mu(x_mu)), scale_factor=2)
        print("Shape after deconv 4_mu:", x_mu.shape)
        x_mu = self.deconv5_mu(x_mu)
        x_mu = F.interpolate(F.leaky_relu(self.bn2d10_mu(x_mu)), scale_factor=2)

        x_mu = self.deconv6_mu(x_mu)       
        print("Shape after deconv 5_mu:", x_mu.shape)
 
        x_mu = torch.sigmoid(x_mu)

        x_alpha = self.deconv4_alpha(x)
        x_alpha = F.interpolate(F.leaky_relu(self.bn2d9_alpha(x_alpha)), scale_factor=2)
        x_alpha = self.deconv5_alpha(x_alpha)        
        x_alpha = F.interpolate(F.leaky_relu(self.bn2d10_alpha(x_alpha)), scale_factor=2)
        x_alpha = self.deconv6_alpha(x_alpha)
        x_alpha = torch.tanh(x_alpha)

        x_beta = self.deconv4_beta(x)
        x_beta = F.interpolate(F.leaky_relu(self.bn2d9_beta(x_beta)), scale_factor=2)
        x_beta = self.deconv5_beta(x_beta)
        x_beta = F.interpolate(F.leaky_relu(self.bn2d10_beta(x_beta)), scale_factor=2)
        x_beta = self.deconv6_beta(x_beta)
        x_beta = torch.tanh(x_beta)/1.6701

        return x_mu, x_alpha, x_beta

class MVTec_net_Autoencoder(BaseNet):

    def __init__(self, rep_dim=500):
        super().__init__()

        self.rep_dim = rep_dim
        self.encoder = MVTec_net(rep_dim=rep_dim)
        self.decoder = MVTec_DecoderNet(rep_dim=rep_dim)

    def forward(self, x, ablation_type='A'):
        x_encoded_mu, x_encoded_alpha, x_encoded_beta = self.encoder(x)
        # sys.exit()
        x_encoded_beta = torch.clamp(x_encoded_beta, np.log(0.1), np.log(2.5))
        standard_beta = torch.ones_like(x_encoded_mu) * 2.0
        # random.seed(0)
        # np.random.seed(0)
        # torch.manual_seed(0)
        # torch.cuda.manual_seed(0)
        # torch.backends.cudnn.deterministic = True

        if ablation_type == 'A':
            sample = sample_ggd(x_encoded_mu, 1e-1 + torch.exp(x_encoded_alpha), 1e-1 + torch.exp(x_encoded_beta))
        elif ablation_type == 'VAE':
            normal_sample = torch.randn(x_encoded_mu.shape).to('cuda:1')
            sample = x_encoded_mu + torch.exp(x_encoded_alpha) * normal_sample
        else:
            sample = x_encoded_mu

        x_recons_mu, x_recons_alpha, x_recons_beta = self.decoder(sample)
        # x_recons_beta = torch.clamp(x_recons_beta, np.log(0.1), np.log(2.5))
        return x_encoded_mu, x_encoded_alpha, x_encoded_beta, x_recons_mu, x_recons_alpha, x_recons_beta, sample