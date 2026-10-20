import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from base.base_net import BaseNet


class FashionMNIST_LeNet(BaseNet):

    def __init__(self, rep_dim=64):
        super().__init__()
        self.rep_dim = rep_dim    
        
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 16, 5, bias=False, padding=2)
        self.bn2d1 = nn.BatchNorm2d(16, eps=1e-04, affine=False)

        self.conv2_mu = nn.Conv2d(16, 32, 5, bias=False, padding=2)
        self.bn2d2_mu = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.fc1_mu = nn.Linear(32 * 7 * 7, 128, bias=False)
        self.bn1d1_mu = nn.BatchNorm1d(128, eps=1e-04, affine=False)
        self.fc2_mu = nn.Linear(128, self.rep_dim, bias=False)

        self.conv2_alpha = nn.Conv2d(16, 32, 5, bias=False, padding=2)
        self.bn2d2_alpha = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.fc1_alpha = nn.Linear(32 * 7 * 7, 128, bias=False)
        self.bn1d1_alpha = nn.BatchNorm1d(128, eps=1e-04, affine=False)
        self.fc2_alpha = nn.Linear(128, self.rep_dim, bias=False)

        self.conv2_beta = nn.Conv2d(16, 32, 5, bias=False, padding=2)
        self.bn2d2_beta = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.fc1_beta = nn.Linear(32 * 7 * 7, 128, bias=False)
        self.bn1d1_beta = nn.BatchNorm1d(128, eps=1e-04, affine=False)
        self.fc2_beta = nn.Linear(128, self.rep_dim, bias=False)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))

        x_mu = self.conv2_mu(x)
        x_mu = self.pool(F.leaky_relu(self.bn2d2_mu(x_mu)))
        x_mu = x_mu.view(int(x_mu.size(0)), -1)
        x_mu = F.leaky_relu(self.bn1d1_mu(self.fc1_mu(x_mu)))
        x_mu = self.fc2_mu(x_mu)

        x_alpha = self.conv2_alpha(x)
        x_alpha = self.pool(F.leaky_relu(self.bn2d2_alpha(x_alpha)))
        x_alpha = x_alpha.view(int(x_alpha.size(0)), -1)
        x_alpha = F.leaky_relu(self.bn1d1_alpha(self.fc1_alpha(x_alpha)))
        x_alpha = self.fc2_alpha(x_alpha)

        x_beta = self.conv2_beta(x)
        x_beta = self.pool(F.leaky_relu(self.bn2d2_beta(x_beta)))
        x_beta = x_beta.view(int(x_beta.size(0)), -1)
        x_beta = F.leaky_relu(self.bn1d1_beta(self.fc1_beta(x_beta)))
        x_beta = self.fc2_beta(x_beta)

        return x_mu , x_alpha, x_beta #x_encoded


class FashionMNIST_LeNet_Decoder(BaseNet):

    def __init__(self, rep_dim=64):
        super().__init__()

        self.rep_dim = rep_dim

        self.fc3 = nn.Linear(self.rep_dim, 128, bias=False)
        self.bn1d2 = nn.BatchNorm1d(128, eps=1e-04, affine=False)
        self.deconv1 = nn.ConvTranspose2d(8, 32, 5, bias=False, padding=2)
        self.bn2d3 = nn.BatchNorm2d(32, eps=1e-04, affine=False)

        self.deconv2_mu = nn.ConvTranspose2d(32, 16, 5, bias=False, padding=3)
        self.bn2d4_mu = nn.BatchNorm2d(16, eps=1e-04, affine=False)
        self.deconv3_mu = nn.ConvTranspose2d(16, 1, 5, bias=False, padding=2)

        self.deconv2_alpha = nn.ConvTranspose2d(32, 16, 5, bias=False, padding=3)
        self.bn2d4_alpha = nn.BatchNorm2d(16, eps=1e-04, affine=False)
        self.deconv3_alpha = nn.ConvTranspose2d(16, 1, 5, bias=False, padding=2)

        self.deconv2_beta = nn.ConvTranspose2d(32, 16, 5, bias=False, padding=3)
        self.bn2d4_beta = nn.BatchNorm2d(16, eps=1e-04, affine=False)
        self.deconv3_beta = nn.ConvTranspose2d(16, 1, 5, bias=False, padding=2)

    def forward(self, x):
        x = self.bn1d2(self.fc3(x))
        x = x.view(int(x.size(0)), int(128 / 16), 4, 4)
        x = F.interpolate(F.leaky_relu(x), scale_factor=2)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn2d3(x)), scale_factor=2)

        x_mu = self.deconv2_mu(x)
        x_mu = F.interpolate(F.leaky_relu(self.bn2d4_mu(x_mu)), scale_factor=2)
        x_mu = self.deconv3_mu(x_mu)
        x_mu = torch.sigmoid(x_mu)

        x_alpha = self.deconv2_alpha(x)
        x_alpha = F.interpolate(F.leaky_relu(self.bn2d4_alpha(x_alpha)), scale_factor=2)
        x_alpha = self.deconv3_alpha(x_alpha)
        x_alpha = torch.sigmoid(x_alpha)

        x_beta = self.deconv2_beta(x)
        x_beta = F.interpolate(F.leaky_relu(self.bn2d4_beta(x_beta)), scale_factor=2)
        x_beta = self.deconv3_beta(x_beta)
        x_beta = torch.sigmoid(x_beta)

        return x_mu , x_alpha, x_beta


class FashionMNIST_LeNet_Autoencoder(BaseNet):

    def __init__(self, rep_dim=64, tau=0.1, delta=0.1, eps=0.1):
        super().__init__()

        self.rep_dim = rep_dim
        self.tau = tau
        self.delta = delta
        self.eps =eps

        self.encoder = FashionMNIST_LeNet(rep_dim=rep_dim)
        self.decoder = FashionMNIST_LeNet_Decoder(rep_dim=rep_dim)

    def forward(self, x, ablation_type:str='A'):

        x_encoded_mu,x_encoded_alpha, x_encoded_beta  = self.encoder(x)
        if(ablation_type=='A'):
            gamma = torch.distributions.gamma.Gamma(1.0/(torch.exp(x_encoded_beta)+self.tau), 1)
            # gamma = torch.distributions.gamma.Gamma(1.0/(x_encoded_beta+self.tau), 1)
            y = gamma.rsample()
            p = torch.ones(x_encoded_mu.shape)*0.5
            p = p.to('cuda' if torch.cuda.is_available() else 'cpu')
            s = torch.bernoulli(p) - 0.5
            sample = x_encoded_mu + 2 *(torch.exp(x_encoded_alpha)+self.delta) * s * (y ** (1.0/(torch.exp(x_encoded_beta)+self.tau)) )
            # sample = x_encoded_mu + 2 *(x_encoded_alpha+self.delta) * s * (y ** (1.0/(x_encoded_beta+self.tau)) )  
                  
        elif(ablation_type=='VAE'):
            normal_sample = torch.randn(x_encoded_mu.shape).to('cuda' if torch.cuda.is_available() else 'cpu')
            sample = x_encoded_mu + torch.exp(x_encoded_alpha) * normal_sample  
                      
        else:
            sample = x_encoded_mu
            
        x_recons_mu, x_recons_alpha, x_recons_beta = self.decoder(sample)   
        return x_encoded_mu,x_encoded_alpha, x_encoded_beta, x_recons_mu, x_recons_alpha, x_recons_beta, sample

