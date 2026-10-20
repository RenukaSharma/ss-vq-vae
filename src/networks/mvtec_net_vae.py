import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet


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

        self.conv5 = nn.Conv2d(256, 512, 3, bias=False, padding=0)
        self.bn2d5 = nn.BatchNorm2d(512, eps=1e-04, affine=False)

        # self.fc1 = nn.Linear(512 * 14 * 14, self.rep_dim, bias=False)
        self.conv6= nn.Conv2d(512, self.rep_dim,1, bias=False)

        self.conv6_alpha= nn.Conv2d(512, self.rep_dim,1, bias=False)

        self.conv6_beta= nn.Conv2d(512, self.rep_dim,1, bias=False)

        # self.bn2d5_1 = nn.BatchNorm2d(self.rep_dim, eps=1e-04, affine=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        x = self.conv4(x)
        x = self.pool(F.leaky_relu(self.bn2d4(x)))
        x = self.conv5(x)
        x = self.pool(F.leaky_relu(self.bn2d5(x)))
        x_mu = self.conv6(x)
        x_alpha = self.tanh(self.conv6_alpha(x))
        x_beta = self.tanh(self.conv6_beta(x))/1.6701
        return x_mu, x_alpha, x_beta

class MVTec_DecoderNet(BaseNet):

    def __init__(self, rep_dim=512):
        super().__init__()

        self.rep_dim = rep_dim
        self.pool = nn.MaxPool2d(2, 2)

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(self.rep_dim , 512, 3, bias=False, padding=0) 
        nn.init.xavier_uniform_(self.deconv1.weight, gain=nn.init.calculate_gain('leaky_relu'))        
        self.bn2d6 = nn.BatchNorm2d(512, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(512, 256, 3, bias=False, padding=0)
        nn.init.xavier_uniform_(self.deconv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d7 = nn.BatchNorm2d(256, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(256, 128, 3, bias=False, padding=0)
        nn.init.xavier_uniform_(self.deconv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d8 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.deconv4 = nn.ConvTranspose2d(128, 64, 3, bias=False, padding=0)
        nn.init.xavier_uniform_(self.deconv4.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d9 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.deconv5 = nn.ConvTranspose2d(64, 32, 3, bias=False, padding=0)
        nn.init.xavier_uniform_(self.deconv5.weight, gain=nn.init.calculate_gain('leaky_relu'))

        self.bn2d10 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.deconv6 = nn.ConvTranspose2d(32, 3, 5, bias=False, padding=0)
        nn.init.xavier_uniform_(self.deconv6.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, x):

        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn2d6(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn2d7(x)), scale_factor=2)
        x = self.deconv3(x)
        x = F.interpolate(F.leaky_relu(self.bn2d8(x)), scale_factor=2)
        x = self.deconv4(x)

        x = F.interpolate(F.leaky_relu(self.bn2d9(x)), scale_factor=2)
        x = self.deconv5(x)

        x = F.interpolate(F.leaky_relu(self.bn2d10(x)), scale_factor=2)

        x = self.deconv6(x)
        x = torch.sigmoid(x)
        return x

class MVTec_net_Autoencoder(BaseNet):

    def __init__(self, rep_dim=512):
        super().__init__()

        self.rep_dim = rep_dim
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder (must match the Deep SVDD network above)
        self.conv1 = nn.Conv2d(3, 32, 5, bias=False, padding=0, stride=2)
        nn.init.xavier_uniform_(self.conv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=0)
        nn.init.xavier_uniform_(self.conv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(64, 128, 5, bias=False, padding=0)
        nn.init.xavier_uniform_(self.conv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)

        self.conv4 = nn.Conv2d(128, 256, 3, bias=False, padding=0)
        nn.init.xavier_uniform_(self.conv4.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d4 = nn.BatchNorm2d(256, eps=1e-04, affine=False)
        
        self.conv5 = nn.Conv2d(256, 512, 3, bias=False, padding=0)
        nn.init.xavier_uniform_(self.conv5.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d5 = nn.BatchNorm2d(512, eps=1e-04, affine=False)

        # self.fc1 = nn.Linear(512 * 14 * 14, self.rep_dim, bias=False)
        self.conv6= nn.Conv2d(512, self.rep_dim,1, bias=False)

        self.bn2d5_1 = nn.BatchNorm2d(self.rep_dim, eps=1e-04, affine=False)

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(self.rep_dim , 512, 3, bias=False, padding=0) 
        nn.init.xavier_uniform_(self.deconv1.weight, gain=nn.init.calculate_gain('leaky_relu'))        
        self.bn2d6 = nn.BatchNorm2d(512, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(512, 256, 3, bias=False, padding=0)
        nn.init.xavier_uniform_(self.deconv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d7 = nn.BatchNorm2d(256, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(256, 128, 3, bias=False, padding=0)
        nn.init.xavier_uniform_(self.deconv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d8 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.deconv4 = nn.ConvTranspose2d(128, 64, 3, bias=False, padding=0)
        nn.init.xavier_uniform_(self.deconv4.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d9 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.deconv5 = nn.ConvTranspose2d(64, 32, 3, bias=False, padding=0)
        nn.init.xavier_uniform_(self.deconv5.weight, gain=nn.init.calculate_gain('leaky_relu'))

        self.bn2d10 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.deconv6 = nn.ConvTranspose2d(32, 3, 5, bias=False, padding=0)
        nn.init.xavier_uniform_(self.deconv6.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, x):

        x = self.conv1(x)
        # print("^^^^^^^^^^after 1st conv,", x.shape)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        # print("^^^^^^^^^^after 1st pool,", x.shape)
        x = self.conv2(x)
        # print("^^^^^^^^^^after 2nd conv,", x.shape)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        # print("^^^^^^^^^^after 2nd pool,", x.shape)
        x = self.conv3(x)
        # print("^^^^^^^^^^after 3rd conv,", x.shape)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        # print("^^^^^^^^^^after 3rd pool,", x.shape)
        x = self.conv4(x)
        # print("^^^^^^^^^^after 3rd conv,", x.shape)
        x = self.pool(F.leaky_relu(self.bn2d4(x)))

        x = self.conv5(x)
        # print("^^^^^^^^^^after 3rd conv,", x.shape)
        x = self.pool(F.leaky_relu(self.bn2d5(x)))

        # x = x.view(x.size(0), -1)
        # print("^^^^^^^^^^after view,", x.shape)
        
        x=self.conv6(x)

        x = self.bn2d5_1(x)
        # x = x.view(x.size(0), int(self.rep_dim / (14 * 14)), 14, 14)
        x = F.leaky_relu(x)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn2d6(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn2d7(x)), scale_factor=2)
        x = self.deconv3(x)
        x = F.interpolate(F.leaky_relu(self.bn2d8(x)), scale_factor=2)
        x = self.deconv4(x)

        x = F.interpolate(F.leaky_relu(self.bn2d9(x)), scale_factor=2)
        x = self.deconv5(x)

        x = F.interpolate(F.leaky_relu(self.bn2d10(x)), scale_factor=2)

        x = self.deconv6(x)
        x = torch.sigmoid(x)
        return x