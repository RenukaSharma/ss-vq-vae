import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from base.base_net import BaseNet
import numpy as np
import logging

class CIFAR10_LeNet(BaseNet):

    def __init__(self, rep_dim=128):
        super().__init__()

        self.rep_dim = rep_dim
        
        self.pool = nn.MaxPool2d(2, 2)
        # for mu
        self.conv1 = nn.Conv2d(3, 32, 3, bias=True, stride=2, padding=0)
        nn.init.xavier_uniform_(self.conv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=True, stride=1, padding=0)
        nn.init.xavier_uniform_(self.conv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=True)

        self.conv3 = nn.Conv2d(64, 128, 3, bias=True, stride=1, padding=0)
        nn.init.xavier_uniform_(self.conv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=True)

        self.conv4 = nn.Conv2d(128, 256, 3, bias=True, stride=1, padding=0)
        nn.init.xavier_uniform_(self.conv4.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d4 = nn.BatchNorm2d(256, eps=1e-04, affine=True)

        self.conv5 = nn.Conv2d(256, 512, 3, bias=True, stride=1, padding=0)
        nn.init.xavier_uniform_(self.conv5.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d5 = nn.BatchNorm2d(512, eps=1e-04, affine=True)

        self.fc1 = nn.Linear(512 * 7 * 7, self.rep_dim, bias=True)
        nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('leaky_relu'))

        self.fc2 = nn.Linear(512 * 7 * 7, self.rep_dim, bias=True)
        nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('leaky_relu'))

        self.fc3 = nn.Linear(512 * 7 * 7, self.rep_dim, bias=True)
        nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, x):
        logger = logging.getLogger()
        x = x.view(-1, 3, 32, 32)
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = F.leaky_relu(self.bn2d1(x))
        x = self.conv2(x)
        # print(x.shape)
        x = F.leaky_relu(self.bn2d2(x))
        x = self.conv3(x)
        # print(x.shape)
        x = F.leaky_relu(self.bn2d3(x))
        x = self.conv4(x)
        # print(x.shape)
        x = F.leaky_relu(self.bn2d4(x))
        x = self.conv5(x)
        # print(x.shape)
        x = F.leaky_relu(self.bn2d5(x))
        x = x.view(int(x.size(0)), -1)
        # print(x.shape)

        # x_mu = torch.sigmoid(self.fc1(x))
        # x_alpha = torch.sigmoid(self.fc2(x))
        # x_beta = torch.sigmoid(self.fc3(x))

        # x_mu = self.fc1(x)
        # x_alpha = self.fc2(x)
        # x_beta = self.fc3(x)

        # x_mu = torch.sigmoid(self.fc1(x))
        # x_alpha = torch.tanh(self.fc2(x))/5
        # x_beta = torch.tanh(self.fc3(x))/1.6701

        # x_mu = torch.sigmoid(self.fc1(x))
        x_mu = (self.fc1(x))
        x_alpha = torch.tanh(self.fc2(x))
        x_beta = torch.tanh(self.fc3(x))/1.6701

        # logger.info("In encoder, alpha min and max are {} and {}".format( str(np.exp(torch.min(x_alpha).item())), str(np.exp(torch.max(x_alpha).item()))))
        # logger.info("In encoder, beta min and max are {} and {}".format( str(np.exp(torch.min(x_beta).item())), str(np.exp(torch.max(x_beta).item()))))

        return x_mu, x_alpha, x_beta #x_encoded

class CIFAR10_LeNet_Decoder(BaseNet):

    def __init__(self, rep_dim=128, tau=0.1, delta=0.1, eps=0.1):
        super().__init__()

        self.rep_dim = rep_dim
        self.tau = tau
        self.delta = delta
        self.eps = eps

        self.d1 = nn.Linear(self.rep_dim, 512*2*7*7, bias=True)
        nn.init.xavier_uniform_(self.d1.weight, gain=nn.init.calculate_gain('leaky_relu'))

        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        # self.pd1 = nn.ReplicationPad2d(1)
        # Conv2d(in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]] = 1, padding: Union[int, Tuple[int, int]] = 0, dilation: Union[int, Tuple[int, int]] = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros')
        self.d2 = nn.Conv2d(512*2, 512, 3, 2, bias=True)
        nn.init.xavier_uniform_(self.d2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn6 = nn.BatchNorm2d(512, 1.e-3, affine=True)

        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        # self.pd2 = nn.ReplicationPad2d(1)
        self.d3 = nn.Conv2d(512, 256, 3, 2, bias=True)
        nn.init.xavier_uniform_(self.d3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn7 = nn.BatchNorm2d(256, 1.e-3, affine=True)

        self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
        # self.pd2 = nn.ReplicationPad2d(1)
        self.d4 = nn.Conv2d(256, 128, 3, 1, bias=True)
        nn.init.xavier_uniform_(self.d4.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn8 = nn.BatchNorm2d(128, 1.e-3, affine=True)

        self.up4 = nn.UpsamplingNearest2d(scale_factor=2)
        # self.pd2 = nn.ReplicationPad2d(1)
        self.d5 = nn.Conv2d(128, 64, 3, 1, 1, bias=True)
        nn.init.xavier_uniform_(self.d5.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn9 = nn.BatchNorm2d(64, 1.e-3, affine=True)

        self.up3_mu = nn.UpsamplingNearest2d(scale_factor=2)
        # self.pd3_mu = nn.ReplicationPad2d(1)
        self.d4_mu = nn.Conv2d(64, 32, 3, 1, 1, bias=True)
        nn.init.xavier_uniform_(self.d4_mu.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn8_mu = nn.BatchNorm2d(32, 1.e-3, affine=True)

        self.d5_mu = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1, bias=True)
        nn.init.xavier_uniform_(self.d5_mu.weight, gain=nn.init.calculate_gain('leaky_relu'))

        self.up3_alpha = nn.UpsamplingNearest2d(scale_factor=2)
        # self.pd3_alpha = nn.ReplicationPad2d(1)
        self.d4_alpha = nn.Conv2d(64, 32, 3, 1, 1, bias=True)
        nn.init.xavier_uniform_(self.d4_alpha.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn8_alpha = nn.BatchNorm2d(32, 1.e-3, affine=True)

        self.d5_alpha = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1, bias=True)
        nn.init.xavier_uniform_(self.d5_alpha.weight, gain=nn.init.calculate_gain('leaky_relu'))

        self.up3_beta = nn.UpsamplingNearest2d(scale_factor=2)
        # self.pd3_beta = nn.ReplicationPad2d(1)
        self.d4_beta = nn.Conv2d(64, 32, 3, 1, 1, bias=True)
        nn.init.xavier_uniform_(self.d4_beta.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn8_beta = nn.BatchNorm2d(32, 1.e-3, affine=True)

        self.d5_beta = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1, bias=True)
        nn.init.xavier_uniform_(self.d5_beta.weight, gain=nn.init.calculate_gain('leaky_relu'))

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        logger = logging.getLogger()

        h1 = F.leaky_relu(self.d1(x))
        # print(h1.shape)
        h1 = h1.view(-1, 512*2 , 7, 7)
        # print(h1.shape)
        h2 = F.leaky_relu(self.bn6(self.d2(self.up1(h1))))
        # print(h2.shape)
        h3 = F.leaky_relu(self.bn7(self.d3(self.up2(h2))))
        # print(h3.shape)
        h4 = F.leaky_relu(self.bn8(self.d4(self.up3(h3))))
        # print(h4.shape)
        h5 = F.leaky_relu(self.bn9(self.d5(self.up4(h4))))
        # print(h5.shape)
        h4_mu = F.leaky_relu(self.bn8_mu(self.d4_mu(self.up3_mu(h5))))
        # print(h4_mu.shape)

        h4_alpha = F.leaky_relu(self.bn8_alpha(self.d4_alpha(self.up3_alpha(h5))))

        h4_beta = F.leaky_relu(self.bn8_beta(self.d4_beta(self.up3_beta(h5))))

        # h5 = self.leakyrelu(self.bn9(self.d5(self.pd4(self.up4(h4)))))

        # x_mu = torch.sigmoid(self.d5_mu(h4_mu))
        # x_alpha = torch.tanh(self.d5_alpha(h4_alpha))/5
        # x_beta = torch.tanh(self.d5_beta(h4_beta))/1.6701

        x_mu = torch.sigmoid(self.d5_mu(h4_mu))
        x_alpha = torch.tanh(self.d5_alpha(h4_alpha))
        x_beta = torch.tanh(self.d5_beta(h4_beta))/1.6701

        # return torch.sigmoid(self.d5_mu(h4_mu)), torch.sigmoid(self.d5_alpha(h4_alpha)), torch.sigmoid(self.d5_beta(h4_beta))
        # return self.d5_mu(h4_mu), self.d5_alpha(h4_alpha), self.d5_beta(h4_beta)
        # logger.info("In DEcoder, alpha min and max are {} and {}".format( str(np.exp(torch.min(x_alpha).item())), str(np.exp(torch.max(x_alpha).item()))))
        # logger.info("In DEcoder, beta min and max are {} and {}".format( str(np.exp(torch.min(x_beta).item())), str(np.exp(torch.max(x_beta).item()))))

        return x_mu, x_alpha, x_beta 
        # return torch.sigmoid(self.d5_mu(h4_mu)), torch.log(torch.sigmoid(self.d5_alpha(h4_alpha))), torch.log(torch.sigmoid(self.d5_beta(h4_beta))*2)
        
        
class CIFAR10_LeNet_Autoencoder(BaseNet):

    def __init__(self, rep_dim=128, tau=0.1, delta=0.1, eps=0.1):
        super().__init__()

        self.rep_dim = rep_dim             
        self.tau = tau
        self.delta = delta
        self.eps =eps

        self.encoder = CIFAR10_LeNet(rep_dim=rep_dim)
        self.decoder = CIFAR10_LeNet_Decoder(rep_dim=rep_dim, tau=tau, delta=delta,  eps=eps)
    
    
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