import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from base.base_net import BaseNet
from torch.distributions.gamma import Gamma


def sample_ggd(mu, alpha, beta):
    gamma = Gamma(1/beta, 1)
    y = gamma.rsample()
    p = torch.ones(mu.shape)*0.5
    s = torch.bernoulli(p) - 0.5
    return mu + 2*alpha*s*(y**(1/beta))


class CIFAR10_LeNet(BaseNet):

    def __init__(self, rep_dim=128):
        super().__init__()

        self.rep_dim = rep_dim

        
        self.pool = nn.MaxPool2d(2, 2)
        # for mu
        self.conv1 = nn.Conv2d(3, 32, 5, bias=False, padding=2)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)

        self.conv3_mu = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        self.bn2d3_mu = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.fc1_mu = nn.Linear(128 * 4 * 4, self.rep_dim, bias=False)

        self.conv3_alpha = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        self.bn2d3_alpha = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.fc1_alpha = nn.Linear(128 * 4 * 4, self.rep_dim, bias=False)

        self.conv3_beta = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        self.bn2d3_beta = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.fc1_beta = nn.Linear(128 * 4 * 4, self.rep_dim, bias=False)

    def forward(self, x):

        # for mu
        x = x.view(-1, 3, 32, 32)
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        
        x_mu = self.conv3_mu(x)
        x_mu = self.pool(F.leaky_relu(self.bn2d3_mu(x_mu)))
        x_mu = x_mu.view(int(x_mu.size(0)), -1)
        x_mu = self.fc1_mu(x_mu)

        x_alpha = self.conv3_alpha(x)
        x_alpha = self.pool(F.leaky_relu(self.bn2d3_alpha(x_alpha)))
        x_alpha = x_alpha.view(int(x_alpha.size(0)), -1)
        x_alpha = torch.tanh(self.fc1_alpha(x_alpha))

        x_beta = self.conv3_beta(x)
        x_beta = self.pool(F.leaky_relu(self.bn2d3_beta(x_beta)))
        x_beta = x_beta.view(int(x_beta.size(0)), -1)
        x_beta = torch.tanh(self.fc1_beta(x_beta))/1.6701

        return x_mu , x_alpha, x_beta  # x_encoded


class CIFAR10_LeNet_Decoder(BaseNet):

    def __init__(self, rep_dim=128):
        super().__init__()

        self.rep_dim = rep_dim

        self.deconv1 = nn.ConvTranspose2d(int(self.rep_dim / (4 * 4)), 128, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d4 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d5 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        
        # for mu
        self.deconv3_mu = nn.ConvTranspose2d(64, 32, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv3_mu.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d6_mu = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.deconv4_mu = nn.ConvTranspose2d(32, 3, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv4_mu.weight, gain=nn.init.calculate_gain('leaky_relu'))
        # for alpha
        self.deconv3_alpha = nn.ConvTranspose2d(64, 32, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv3_alpha.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d6_alpha = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.deconv4_alpha = nn.ConvTranspose2d(32, 3, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv4_alpha.weight, gain=nn.init.calculate_gain('leaky_relu'))
        # for beta
        self.deconv3_beta = nn.ConvTranspose2d(64, 32, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv3_beta.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d6_beta = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.deconv4_beta = nn.ConvTranspose2d(32, 3, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv4_beta.weight, gain=nn.init.calculate_gain('leaky_relu'))


    def forward(self, x):
        
        x = x.view(int(x.size(0)), int(self.rep_dim / (4 * 4)), 4, 4)
        x = F.leaky_relu(x)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn2d4(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn2d5(x)), scale_factor=2)

        # for mu
        x_mu = self.deconv3_mu(x)
        x_mu = F.interpolate(F.leaky_relu(self.bn2d6_mu(x_mu)), scale_factor=2)
        x_mu = self.deconv4_mu(x_mu)
        x_mu = torch.sigmoid(x_mu)
        # for alpha
        x_alpha = self.deconv3_alpha(x)
        x_alpha = F.interpolate(F.leaky_relu(self.bn2d6_alpha(x_alpha)), scale_factor=2)
        x_alpha = self.deconv4_alpha(x_alpha)
        x_alpha = torch.tanh(x_alpha)
        # for beta
        x_beta = self.deconv3_beta(x)
        x_beta = F.interpolate(F.leaky_relu(self.bn2d6_beta(x_beta)), scale_factor=2)
        x_beta = self.deconv4_beta(x_beta)
        x_beta = torch.tanh(x_beta)/1.6701

        return x_mu , x_alpha, x_beta  # x_decoded or reconstructed


class CIFAR10_LeNet_Autoencoder(BaseNet):

    def __init__(self, rep_dim=128, tau=0.1, delta=0.1, eps=0.1):
        super().__init__()

        self.rep_dim = rep_dim            
        self.tau = tau
        self.delta = delta
        self.eps =eps

        self.encoder = CIFAR10_LeNet(rep_dim=rep_dim)
        self.decoder = CIFAR10_LeNet_Decoder(rep_dim=rep_dim)

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
