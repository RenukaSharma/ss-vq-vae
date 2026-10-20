import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

from torch.autograd import Variable
from base.base_net import BaseNet
from torch.distributions.gamma import Gamma
from torch.distributions.normal import Normal


def sample_ggd(mu, alpha, beta):
    gamma = Gamma(1/beta, 1)
    y = gamma.rsample()
    p = torch.ones_like(mu)
    s = torch.bernoulli(p) - 0.5
    return mu + 2*alpha*s*(y**(1/beta))


class Nanofibre_LeNet(BaseNet):

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

        self.fc2 = nn.Linear(128 * 4 * 4, self.rep_dim, bias=False)
        nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('leaky_relu'))

        self.fc3 = nn.Linear(128 * 4 * 4, self.rep_dim, bias=False)
        nn.init.xavier_uniform_(self.fc3.weight, gain=nn.init.calculate_gain('leaky_relu'))

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        # super().__init__()
        #
        # self.rep_dim = rep_dim
        # self.pool = nn.MaxPool2d(2, 2)
        #
        # self.conv1 = nn.Conv2d(3, 32, 5, bias=False, padding=2)
        # self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        # self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        # self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        # self.conv3 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        # self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        # self.fc1 = nn.Linear(128 * 4 * 4, self.rep_dim, bias=False)

    def forward(self, x):

        x = x.view(-1, 1, 32, 32)
        x = self.conv1(x)
        x = F.leaky_relu(self.bn2d1(x))
        x = self.conv2(x)
        x = F.leaky_relu(self.bn2d2(x))
        x = self.conv3(x)
        x = F.leaky_relu(self.bn2d3(x))
        x = x.view(int(x.size(0)), -1)

        x_mu = 2*self.sigmoid(self.fc1(x)) - 1
        x_alpha = 2*self.sigmoid(self.fc2(x)) - 1
        x_beta = 2*self.sigmoid(self.fc3(x)) - 1

        # x_mu = self.tanh(self.fc1(x))
        # x_alpha = self.tanh(self.fc2(x))
        # x_beta = self.tanh(self.fc3(x))/1.6701

        return x_mu, torch.exp(x_alpha), torch.exp(x_beta)  # x_encoded

        

class Nanofibre_LeNet_Decoder(BaseNet):

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

        self.up3_alpha = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd3_alpha = nn.ReplicationPad2d(1)
        self.d4_alpha = nn.Conv2d(64, 32, 3, 1)
        nn.init.xavier_uniform_(self.d4_alpha.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn8_alpha = nn.BatchNorm2d(32, 1.e-3)

        self.d5_alpha = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.d5_alpha.weight, gain=nn.init.calculate_gain('leaky_relu'))

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

        h4_alpha = self.leakyrelu(self.bn8_alpha(self.d4_alpha(self.pd3_alpha(self.up3_alpha(h3)))))

        h4_beta = self.leakyrelu(self.bn8_beta(self.d4_beta(self.pd3_beta(self.up3_beta(h3)))))
       
        x_mu = self.sigmoid(self.d5_mu(h4_mu))

        x_alpha = 2*self.sigmoid(self.d5_alpha(h4_alpha)) - 1
        x_beta = 2*self.sigmoid(self.d5_beta(h4_beta)) - 1

        # x_alpha = self.tanh(self.d5_alpha(h4_alpha))
        # x_beta = self.tanh(self.d5_beta(h4_beta))/1.6701

        return x_mu, torch.exp(x_alpha), torch.exp(x_beta)


class Nanofibre_LeNet_Autoencoder(BaseNet):

    def __init__(self, rep_dim=128):
        super().__init__()

        self.rep_dim = rep_dim             
        
        self.encoder = Nanofibre_LeNet(rep_dim=rep_dim)
        self.decoder = Nanofibre_LeNet_Decoder(rep_dim=rep_dim)

    def forward(self, x, ablation_type='A'):
        x_encoded_mu, x_encoded_alpha, x_encoded_beta = self.encoder(x)
        # x_encoded_beta = torch.clamp(x_encoded_beta, np.log(0.1), np.log(2.5))
        # standard_beta = torch.ones_like(x_encoded_mu) * 2.0
        # random.seed(0)
        # np.random.seed(0)
        # torch.manual_seed(0)
        # torch.cuda.manual_seed(0)
        # torch.backends.cudnn.deterministic = True

        if ablation_type == 'A':
            sample = sample_ggd(x_encoded_mu, 1e-6 + (x_encoded_alpha), 1e-6 + (x_encoded_beta))
        elif ablation_type == 'VAE':
            normal_sample = torch.randn(x_encoded_mu.shape).to('cuda:1')
            sample = x_encoded_mu + (x_encoded_alpha) * normal_sample
        else:
            sample = x_encoded_mu

        x_recons_mu, x_recons_alpha, x_recons_beta = self.decoder(sample)
        # x_recons_beta = torch.clamp(x_recons_beta, np.log(0.1), np.log(2.5))
        return x_encoded_mu, x_encoded_alpha, x_encoded_beta, x_recons_mu, x_recons_alpha, x_recons_beta, sample


# print("Sample Norm: ", torch.max(sample), torch.min(sample))
        # print("Mu", torch.mean(x_encoded_mu))
        # print("Alphs: ", torch.mean(torch.exp(x_encoded_alpha)))
        # print("Beta: ", torch.mean(torch.exp(x_encoded_beta)), torch.max(torch.exp(x_encoded_beta)), torch.min(torch.exp(x_encoded_beta)))





# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from torch.autograd import Variable
# from base.base_net import BaseNet
# import numpy as np
# import random
# import logging


# class Nanofibre_LeNet(BaseNet):
#     def __init__(self, rep_dim=128):
#         super().__init__()

#         self.rep_dim = rep_dim

#         # for mu
#         self.conv1 = nn.Conv2d(1, 8, 4, bias=True, stride=2, padding=1)
#         nn.init.xavier_uniform_(self.conv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
#         self.bn2d1 = nn.BatchNorm2d(8, eps=1e-04, affine=True)

#         self.conv2 = nn.Conv2d(8, 16, 4, bias=True, stride=2, padding=1)
#         nn.init.xavier_uniform_(self.conv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
#         self.bn2d2 = nn.BatchNorm2d(16, eps=1e-04, affine=True)

#         self.conv3 = nn.Conv2d(16, 32, 4, bias=True, stride=2, padding=1)
#         nn.init.xavier_uniform_(self.conv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
#         self.bn2d3 = nn.BatchNorm2d(32, eps=1e-04, affine=True)

#         self.fc1 = nn.Linear(32 * 3 * 3, self.rep_dim, bias=True)
#         nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('leaky_relu'))

#         self.fc2 = nn.Linear(32 * 3 * 3, self.rep_dim, bias=True)
#         nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('leaky_relu'))

#         self.fc3 = nn.Linear(32 * 3 * 3, self.rep_dim, bias=True)
#         nn.init.xavier_uniform_(self.fc3.weight, gain=nn.init.calculate_gain('leaky_relu'))

#     def forward(self, x):

#         logger = logging.getLogger()

#         x = x.view(-1, 1, 28, 28)
#         x = self.conv1(x)
#         x = F.leaky_relu(self.bn2d1(x))
#         x = self.conv2(x)
#         x = F.leaky_relu(self.bn2d2(x))
#         x = self.conv3(x)
#         x = F.leaky_relu(self.bn2d3(x))
#         x = x.view(int(x.size(0)), -1)

#         # x_mu = torch.sigmoid(self.fc1(x))
#         # x_alpha = torch.tanh(self.fc2(x))/5
#         # x_beta = torch.tanh(self.fc3(x))/1.6701

#         # x_mu = torch.sigmoid(self.fc1(x))
#         x_mu = (self.fc1(x))
#         x_alpha = 2*torch.sigmoid(self.fc2(x))-1
#         x_beta = 2*torch.sigmoid(self.fc3(x))-1

#         # x_mu = (self.fc1(x))
#         # x_alpha = torch.tanh(self.fc2(x))
#         # x_beta = torch.tanh(self.fc3(x))
        
#         return x_mu, torch.exp(x_alpha), torch.exp(x_beta) #x_encoded


# class Nanofibre_LeNet_Decoder(BaseNet):

#     def __init__(self, rep_dim=128, tau=0.1, delta=0.1):
#         super().__init__()

#         self.rep_dim = rep_dim
#         self.tau = tau
#         self.delta = delta

#         self.d1 = nn.Linear(self.rep_dim, 32*2*3*3, bias=True)
#         nn.init.xavier_uniform_(self.d1.weight, gain=nn.init.calculate_gain('leaky_relu'))

#         self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
#         self.pd1 = nn.ReplicationPad2d(1)
#         self.d2 = nn.Conv2d(32*2, 32, 3, 1, bias=True)
#         nn.init.xavier_uniform_(self.d2.weight, gain=nn.init.calculate_gain('leaky_relu'))
#         self.bn6 = nn.BatchNorm2d(32, 1.e-3, affine=True)

#         self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
#         self.pd2 = nn.ReplicationPad2d(2)
#         self.d3 = nn.Conv2d(32, 16, 3, 1, bias=True)
#         nn.init.xavier_uniform_(self.d3.weight, gain=nn.init.calculate_gain('leaky_relu'))
#         self.bn7 = nn.BatchNorm2d(16, 1.e-3, affine=True)

#         self.up3_mu = nn.UpsamplingNearest2d(scale_factor=2)
#         self.pd3_mu = nn.ReplicationPad2d(1)
#         self.d4_mu = nn.Conv2d(16, 8, 3, 1, bias=True)
#         nn.init.xavier_uniform_(self.d4_mu.weight, gain=nn.init.calculate_gain('leaky_relu'))
#         self.bn8_mu = nn.BatchNorm2d(8, 1.e-3, affine=True)

#         self.d5_mu = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1, bias=True)
#         nn.init.xavier_uniform_(self.d5_mu.weight, gain=nn.init.calculate_gain('leaky_relu'))

#         self.up3_alpha = nn.UpsamplingNearest2d(scale_factor=2)
#         self.pd3_alpha = nn.ReplicationPad2d(1)
#         self.d4_alpha = nn.Conv2d(16, 8, 3, 1, bias=True)
#         nn.init.xavier_uniform_(self.d4_alpha.weight, gain=nn.init.calculate_gain('leaky_relu'))
#         self.bn8_alpha = nn.BatchNorm2d(8, 1.e-3, affine=True)

#         self.d5_alpha = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1, bias=True)
#         nn.init.xavier_uniform_(self.d5_alpha.weight, gain=nn.init.calculate_gain('leaky_relu'))

#         self.up3_beta = nn.UpsamplingNearest2d(scale_factor=2)
#         self.pd3_beta = nn.ReplicationPad2d(1)
#         self.d4_beta = nn.Conv2d(16, 8, 3, 1, bias=True)
#         nn.init.xavier_uniform_(self.d4_beta.weight, gain=nn.init.calculate_gain('leaky_relu'))
#         self.bn8_beta = nn.BatchNorm2d(8, 1.e-3, affine=True)

#         self.d5_beta = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1, bias=True)
#         nn.init.xavier_uniform_(self.d5_beta.weight, gain=nn.init.calculate_gain('leaky_relu'))

#         self.leakyrelu = nn.LeakyReLU(0.2)
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
        
#         logger = logging.getLogger()
#         h1 = F.leaky_relu(self.d1(x))
#         h1 = h1.view(-1,32*2 , 3, 3)
#         # print("hue", h1.shape)
#         h2 = F.leaky_relu(self.bn6(self.d2(self.pd1(self.up1(h1)))))
#         # print(h2.shape)
#         h3 = F.leaky_relu(self.bn7(self.d3(self.pd2(self.up2(h2)))))
#         # print(h3.shape)
#         h4_mu = F.leaky_relu(self.bn8_mu(self.d4_mu(self.pd3_mu(self.up3_mu(h3)))))

#         h4_alpha = F.leaky_relu(self.bn8_alpha(self.d4_alpha(self.pd3_alpha(self.up3_alpha(h3)))))

#         h4_beta = F.leaky_relu(self.bn8_beta(self.d4_beta(self.pd3_beta(self.up3_beta(h3)))))

#         # h5 = self.leakyrelu(self.bn9(self.d5(self.pd4(self.up4(h4)))))

#         # x_mu = torch.sigmoid(self.d5_mu(h4_mu))
#         # x_alpha = torch.tanh(self.d5_alpha(h4_alpha))/5
#         # x_beta = torch.tanh(self.d5_beta(h4_beta))/1.6701

#         x_mu = torch.sigmoid(self.d5_mu(h4_mu))
#         x_alpha = 2*torch.sigmoid(self.d5_alpha(h4_alpha))-1
#         x_beta = 2*torch.sigmoid(self.d5_beta(h4_beta))-1

#         # x_mu = torch.sigmoid(self.d5_mu(h4_mu))
#         # x_alpha = torch.tanh(self.d5_alpha(h4_alpha))
#         # x_beta = torch.tanh(self.d5_beta(h4_beta))
        

#         return x_mu, torch.exp(x_alpha), torch.exp(x_beta) #decoded


# class Nanofibre_LeNet_Autoencoder(BaseNet):

#     def __init__(self, rep_dim=32, tau=0.1, delta=0.1):
#         super().__init__()

#         self.rep_dim = rep_dim
#         self.tau = tau
#         self.delta = delta
        
#         self.encoder = Nanofibre_LeNet(rep_dim=rep_dim)
#         self.decoder = Nanofibre_LeNet_Decoder(rep_dim=rep_dim, tau=tau, delta=delta)

    
#     def forward(self, x, ablation_type:str='A'):
        
#         x_encoded_mu,x_encoded_alpha, x_encoded_beta  = self.encoder(x)
#         # print("x_encoded_mu: ", x_encoded_mu.shape)
#         if(ablation_type=='A'):
#             gamma = torch.distributions.gamma.Gamma(1.0/((x_encoded_beta)+self.tau), 1)
#             # gamma = torch.distributions.gamma.Gamma(1.0/(x_encoded_beta+self.tau), 1)
#             y = gamma.rsample()
#             p = torch.ones_like(x_encoded_mu)
#             # p = torch.ones(x_encoded_mu.shape)*0.5
#             # p = p.to('cuda' if torch.cuda.is_available() else 'cpu')
#             s = torch.bernoulli(p) - 0.5
#             sample = x_encoded_mu + 2 *((x_encoded_alpha)+self.delta) * s * (y ** (1.0/((x_encoded_beta)+self.tau)))
#             # sample = x_encoded_mu + 2 *(x_encoded_alpha+self.delta) * s * (y ** (1.0/(x_encoded_beta+self.tau)) )  
                  
#         elif(ablation_type=='VAE'):
#             normal_sample = torch.randn(x_encoded_mu.shape).to('cuda' if torch.cuda.is_available() else 'cpu')
#             sample = x_encoded_mu + torch.exp(x_encoded_alpha) * normal_sample  

#         else:
#             sample = x_encoded_mu
            
#         x_recons_mu, x_recons_alpha, x_recons_beta = self.decoder(sample)   
#         return x_encoded_mu,x_encoded_alpha, x_encoded_beta, x_recons_mu, x_recons_alpha, x_recons_beta, sample