import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from base.base_net import BaseNet


class MNIST_LeNet(BaseNet):

    def __init__(self, rep_dim=32):
        super().__init__()

        self.rep_dim = rep_dim           
        
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)

        self.conv2_mu = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2_mu = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.fc1_mu = nn.Linear(4 * 7 * 7, self.rep_dim, bias=False)

        # self.conv2_alpha = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        # self.bn2_alpha = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        # self.fc1_alpha = nn.Linear(4 * 7 * 7, self.rep_dim, bias=False)

        # self.conv2_beta = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        # self.bn2_beta = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        # self.fc1_beta = nn.Linear(4 * 7 * 7, self.rep_dim, bias=False)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))

        x_mu = self.conv2_mu(x)
        x_mu = self.pool(F.leaky_relu(self.bn2_mu(x_mu)))
        x_mu = x_mu.view(int(x_mu.size(0)), -1)
        x_mu = self.fc1_mu(x_mu)

        # x_alpha = self.conv2_alpha(x)
        # x_alpha = self.pool(F.leaky_relu(self.bn2_alpha(x_alpha)))
        # x_alpha = x_alpha.view(int(x_alpha.size(0)), -1)
        # x_alpha = self.fc1_alpha(x_alpha)

        # x_beta = self.conv2_beta(x)
        # x_beta = self.pool(F.leaky_relu(self.bn2_beta(x_beta)))
        # x_beta = x_beta.view(int(x_beta.size(0)), -1)
        # x_beta = self.fc1_beta(x_beta)

        return x_mu #, x_alpha, x_beta #x_encoded


class MNIST_LeNet_Decoder(BaseNet):

    def __init__(self, rep_dim=32):
        super().__init__()

        self.rep_dim = rep_dim

        # Decoder network
        self.deconv1 = nn.ConvTranspose2d(2, 4, 5, bias=False, padding=2)
        self.bn3 = nn.BatchNorm2d(4, eps=1e-04, affine=False)

        self.deconv2_mu = nn.ConvTranspose2d(4, 8, 5, bias=False, padding=3)
        self.bn4_mu = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.deconv3_mu = nn.ConvTranspose2d(8, 1, 5, bias=False, padding=2)

        # self.deconv2_alpha = nn.ConvTranspose2d(4, 8, 5, bias=False, padding=3)
        # self.bn4_alpha = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        # self.deconv3_alpha = nn.ConvTranspose2d(8, 1, 5, bias=False, padding=2)

        # self.deconv2_beta = nn.ConvTranspose2d(4, 8, 5, bias=False, padding=3)
        # self.bn4_beta = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        # self.deconv3_beta = nn.ConvTranspose2d(8, 1, 5, bias=False, padding=2)

    def forward(self, x):
        x = x.view(int(x.size(0)), int(self.rep_dim / 16), 4, 4)
        x = F.interpolate(F.leaky_relu(x), scale_factor=2)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn3(x)), scale_factor=2)

        x_mu = self.deconv2_mu(x)
        x_mu = F.interpolate(F.leaky_relu(self.bn4_mu(x_mu)), scale_factor=2)
        x_mu = self.deconv3_mu(x_mu)
        x_mu = torch.sigmoid(x_mu)

        # x_alpha = self.deconv2_alpha(x)
        # x_alpha = F.interpolate(F.leaky_relu(self.bn4_alpha(x_alpha)), scale_factor=2)
        # x_alpha = self.deconv3_alpha(x_alpha)
        # x_alpha = torch.sigmoid(x_alpha)

        # x_beta = self.deconv2_beta(x)
        # x_beta = F.interpolate(F.leaky_relu(self.bn4_beta(x_beta)), scale_factor=2)
        # x_beta = self.deconv3_beta(x_beta)
        # x_beta = torch.sigmoid(x_beta)

        return x_mu #, x_alpha, x_beta


class MNIST_LeNet_Autoencoder(BaseNet):

    def __init__(self, rep_dim=32):
        super().__init__()

        self.rep_dim = rep_dim
        
        self.encoder = MNIST_LeNet(rep_dim=rep_dim)
        self.decoder = MNIST_LeNet_Decoder(rep_dim=rep_dim)

    def forward(self, x):
        
        x_encoded_mu  = self.encoder(x)
        x_recons_mu = self.decoder(x_encoded_mu)
        return x_encoded_mu, x_recons_mu
