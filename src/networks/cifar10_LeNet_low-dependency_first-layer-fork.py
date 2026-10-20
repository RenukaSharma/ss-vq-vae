import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from base.base_net import BaseNet


class CIFAR10_LeNet(BaseNet):

    def __init__(self, rep_dim=128):
        super().__init__()

        self.rep_dim = rep_dim
        
        # self.p = torch.nn.Parameter(Variable(torch.ones(128, self.rep_dim), requires_grad=True))
        # self.c = torch.nn.Parameter(Variable(torch.ones(128, self.rep_dim), requires_grad=True))

        
        self.pool = nn.MaxPool2d(2, 2)
        # for mu
        self.conv1_mu = nn.Conv2d(3, 32, 5, bias=False, padding=2)
        self.bn2d1_mu = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2_mu = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        self.bn2d2_mu = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3_mu = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        self.bn2d3_mu = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.fc1_mu = nn.Linear(128 * 4 * 4, self.rep_dim, bias=False)

        # for alpha
        self.conv1_alpha = nn.Conv2d(3, 32, 5, bias=False, padding=2)
        self.bn2d1_alpha = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2_alpha = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        self.bn2d2_alpha = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3_alpha = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        self.bn2d3_alpha = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.fc1_alpha = nn.Linear(128 * 4 * 4, self.rep_dim, bias=False)

        # for beta
        self.conv1_beta = nn.Conv2d(3, 32, 5, bias=False, padding=2)
        self.bn2d1_beta = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2_beta = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        self.bn2d2_beta = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3_beta = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        self.bn2d3_beta = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.fc1_beta = nn.Linear(128 * 4 * 4, self.rep_dim, bias=False)

    def forward(self, x):

        # for mu
        x_mu = x.view(-1, 3, 32, 32)
        x_mu = self.conv1_mu(x)
        x_mu = self.pool_mu(F.leaky_relu(self.bn2d1_mu(x)))
        x_mu = self.conv2_mu(x)
        x_mu = self.pool(F.leaky_relu(self.bn2d2_mu(x)))
        x_mu = self.conv3_mu(x)
        x_mu = self.pool(F.leaky_relu(self.bn2d3_mu(x)))
        x_mu = x.view(int(x.size(0)), -1)
        x_mu = self.fc1_mu(x)

        # for alpha
        x_alpha = x.view(-1, 3, 32, 32)
        x_alpha = self.conv1_alpha(x)
        x_alpha = self.pool(F.leaky_relu(self.bn2d1_alpha(x)))
        x_alpha = self.conv2_alpha(x)
        x_alpha = self.pool(F.leaky_relu(self.bn2d2_alpha(x)))
        x_alpha = self.conv3_alpha(x)
        x_alpha = self.pool(F.leaky_relu(self.bn2d3_alpha(x)))
        x_alpha = x.view(int(x.size(0)), -1)
        x_alpha = self.fc1_alpha(x)

        # for beta
        x_beta = x.view(-1, 3, 32, 32)
        x_beta = self.conv1_beta(x)
        x_beta = self.pool(F.leaky_relu(self.bn2d1_beta(x)))
        x_beta = self.conv2_beta(x)
        x_beta = self.pool(F.leaky_relu(self.bn2d2_beta(x)))
        x_beta = self.conv3_beta(x)
        x_beta = self.pool(F.leaky_relu(self.bn2d3_beta(x)))
        x_beta = x.view(int(x.size(0)), -1)
        x_beta = self.fc1_beta(x)

        return x_mu, x_alpha, x_beta #x_encoded


class CIFAR10_LeNet_Decoder(BaseNet):

    def __init__(self, rep_dim=128):
        super().__init__()

        self.rep_dim = rep_dim

        # for mu
        self.deconv1_mu = nn.ConvTranspose2d(int(self.rep_dim / (4 * 4)), 128, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv1_mu.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d4_mu = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.deconv2_mu = nn.ConvTranspose2d(128, 64, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv2_mu.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d5_mu = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.deconv3_mu = nn.ConvTranspose2d(64, 32, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv3_mu.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d6_mu = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.deconv4_mu = nn.ConvTranspose2d(32, 3, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv4_mu.weight, gain=nn.init.calculate_gain('leaky_relu'))

        # for alpha
        self.deconv1_alpha = nn.ConvTranspose2d(int(self.rep_dim / (4 * 4)), 128, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv1_alpha.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d4_alpha = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.deconv2_alpha = nn.ConvTranspose2d(128, 64, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv2_alpha.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d5_alpha = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.deconv3_alpha = nn.ConvTranspose2d(64, 32, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv3_alpha.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d6_alpha = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.deconv4_alpha = nn.ConvTranspose2d(32, 3, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv4_alpha.weight, gain=nn.init.calculate_gain('leaky_relu'))

        # for beta
        self.deconv1_beta = nn.ConvTranspose2d(int(self.rep_dim / (4 * 4)), 128, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv1_beta.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d4_beta = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.deconv2_beta = nn.ConvTranspose2d(128, 64, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv2_beta.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d5_beta = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.deconv3_beta = nn.ConvTranspose2d(64, 32, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv3_beta.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d6_beta = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.deconv4_beta = nn.ConvTranspose2d(32, 3, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv4_beta.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, x):

        # for mu
        x = x.view(int(x.size(0)), int(self.rep_dim / (4 * 4)), 4, 4)
        x = F.leaky_relu(x)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn2d4(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn2d5(x)), scale_factor=2)
        x = self.deconv3(x)
        x = F.interpolate(F.leaky_relu(self.bn2d6(x)), scale_factor=2)
        x = self.deconv4(x)
        x = torch.sigmoid(x)

        # for alpha
        x = x.view(int(x.size(0)), int(self.rep_dim / (4 * 4)), 4, 4)
        x = F.leaky_relu(x)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn2d4(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn2d5(x)), scale_factor=2)
        x = self.deconv3(x)
        x = F.interpolate(F.leaky_relu(self.bn2d6(x)), scale_factor=2)
        x = self.deconv4(x)
        x = torch.sigmoid(x)

        # for mu
        x = x.view(int(x.size(0)), int(self.rep_dim / (4 * 4)), 4, 4)
        x = F.leaky_relu(x)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn2d4(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn2d5(x)), scale_factor=2)
        x = self.deconv3(x)
        x = F.interpolate(F.leaky_relu(self.bn2d6(x)), scale_factor=2)
        x = self.deconv4(x)
        x = torch.sigmoid(x)

        return x #x_decoded


class CIFAR10_LeNet_Autoencoder(BaseNet):

    def __init__(self, rep_dim=128):
        super().__init__()

        self.rep_dim = rep_dim
        """
        self.p = torch.nn.Parameter(torch.tensor(1.0))
        self.c = torch.nn.Parameter(torch.tensor(1.0))
        """
        """
        self.p =  torch.ones_like(32, 32)
        self.c =  torch.ones_like(32, 32)
        """
        """
        self.p = torch.nn.Parameter(Variable(torch.tensor(1.0), requires_grad=True))
        self.c = torch.nn.Parameter(Variable(torch.tensor(1.0), requires_grad=True))
        """
        
        # self.p = torch.nn.Parameter(Variable(torch.ones(128, 3, 32,32), requires_grad=True))
        # self.c = torch.nn.Parameter(Variable(torch.ones(128, 3, 32,32), requires_grad=True))
        
        
        self.encoder = CIFAR10_LeNet(rep_dim=rep_dim)
        self.decoder = CIFAR10_LeNet_Decoder(rep_dim=rep_dim)

    def forward(self, x):
        x_encoded = self.encoder(x)
        x_recons = self.decoder(x_encoded)
        return x_recons
