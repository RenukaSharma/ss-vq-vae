from torch.utils.data import Subset,random_split,ConcatDataset
from PIL import Image
from torch import randperm
from torchvision.datasets import CIFAR10
from base.torchvision_dataset import TorchvisionDataset
# from .preprocessing import get_target_label_idx, global_contrast_normalization
from .preprocessing import create_semisupervised_setting

import numpy as np
import cv2
import torchvision.transforms as transforms
import random
import torch.utils.data as data
from torchvision.datasets import ImageFolder
import os
import copy
from PIL import ImageFile
import torch
import logging

from torch.distributions.gamma import Gamma
from torch.distributions.uniform import Uniform

def sample_ggd(mu, alpha, beta):
    gamma = Gamma(1/beta, 1)
    y = gamma.rsample()
    p = torch.ones(mu.shape)*0.5
    s = torch.bernoulli(p) - 0.5
    return mu + 2*alpha*s*(y**(1/beta))

ImageFile.LOAD_TRUNCATED_IMAGES=True

class MVTec_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class: int = 0, known_outlier_class: int = 1, n_known_outlier_classes: int = 1,
                 ratio_known_normal: float = 0.0, ratio_known_outlier: float = 0.0, ratio_pollution: float = 0.0, length: int=5000, category='carpet'):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier
        
        self.normal_classes = 0
        self.outlier_classes = 1

        self.normal_classes = tuple([self.normal_classes])
        self.outlier_classes = tuple([self.outlier_classes])

        if n_known_outlier_classes == 0:
            self.known_outlier_classes = ()
        elif n_known_outlier_classes == 1:
            self.known_outlier_classes = tuple([known_outlier_class])
        else:
            self.known_outlier_classes = tuple(random.sample(self.outlier_classes, n_known_outlier_classes))
              
        # transform = None

        # transform = transforms.ToTensor()
        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes)) #if it lies in outlier_classes then 1, 0 otherwise : transform to be applied on target
        normal_class_name=category

        transform = transforms.Compose([ # transforms.Resize((256, 256)),
                                transforms.Resize((32,32)),
                                # transforms.RandomCrop((32,32)),
                                transforms.ToTensor()
                                ])
        
        transform_2 = transforms.Compose([ # transforms.Resize((256, 256)),
                                transforms.RandomApply([ transforms.RandomAffine(degrees= 30)],p=1),
                                transforms.Resize((32,32)),
                                # transforms.RandomCrop((32,32)),
                                transforms.ToTensor()
                                ])
        transform_3 = transforms.Compose([ # transforms.Resize((256, 256)),
                                transforms.RandomApply([ transforms.RandomAffine(degrees= 45)],p=1),
                                transforms.Resize((32,32)),
                                # transforms.RandomCrop((32,32)),
                                transforms.ToTensor()
                                ])

        train_set = MyMVTec(root=self.root+'/train', train=True, transform=transform, target_transform=target_transform,
                              download=True)
        train_set_2 = MyMVTec(root=self.root+'/train', train=True, transform=transform_2, target_transform=target_transform,
                              download=True)
        train_set_3 = MyMVTec(root=self.root+'/train', train=True, transform=transform_3, target_transform=target_transform,
                              download=True)
        print("The length of train_set is",len(train_set))
        print("The length of train_set_2 is",len(train_set_2))
        print("The length of train_set_3 is",len(train_set_3))

        # Create semi-supervised setting
        datasets =[]
        idx, _, semi_targets = create_semisupervised_setting(np.array(train_set.targets), self.normal_classes,
                                                             self.outlier_classes, self.known_outlier_classes,
                                                             ratio_known_normal, ratio_known_outlier, ratio_pollution)
        train_set.semi_targets[idx] = torch.tensor(semi_targets)[:train_set.semi_targets[idx].shape[0]]  # set respective semi-supervised labels
        datasets.append(Subset(train_set, idx))

        # import ipdb; ipdb.set_trace()

        idx, _, semi_targets = create_semisupervised_setting(np.array(train_set_2.targets), self.normal_classes,
                                                             self.outlier_classes, self.known_outlier_classes,
                                                             ratio_known_normal, ratio_known_outlier, ratio_pollution)
        train_set_2.semi_targets[idx] = torch.tensor(semi_targets)[:train_set_2.semi_targets[idx].shape[0]]  # set respective semi-supervised labels
        datasets.append(Subset(train_set_2, idx))

        idx, _, semi_targets = create_semisupervised_setting(np.array(train_set_3.targets), self.normal_classes,
                                                             self.outlier_classes, self.known_outlier_classes,
                                                             ratio_known_normal, ratio_known_outlier, ratio_pollution)
        train_set_3.semi_targets[idx] = torch.tensor(semi_targets)[:train_set_3.semi_targets[idx].shape[0]]  # set respective semi-supervised labels
        datasets.append(Subset(train_set_3, idx))

        # Subset train_set to semi-supervised setup
        self.train_set = ConcatDataset (datasets)
        
        logger = logging.getLogger()
        logger.info('The length of train_set is %d' % len(self.train_set))

        
                    
        # self.test_set = ConcatDataset   (
        #                                 [MyMVTec(root=self.root+'/test', train=False, transform=transform, 
        #                                 target_transform=target_transform, download=True),
        #                                 MyMVTec(root=self.root+'/test', train=False, transform=transform_2, 
        #                                 target_transform=target_transform, download=True),
        #                                 MyMVTec(root=self.root+'/test', train=False, transform=transform_3, 
        #                                 target_transform=target_transform, download=True)
        #                                 ]
        #                                 )
        self.test_set = MyMVTec(root=self.root+'/test', train=False, transform=transform, 
                                        target_transform=target_transform, download=True)
        #########################
        

class MyMVTec(ImageFolder):#MyCIFAR10(CIFAR10):
    """Torchvision CIFAR10 class with patch of __getitem__ method to also return the index of a data sample."""


    def __init__(self,root, train=False, transform=None, target_transform=None, download=False):
        super(MyMVTec, self).__init__(root, transform=transform, target_transform=target_transform)
        #list of (path, class_to_idx[target]) = samples
        
        self.train=train # training set or test set
        self.transform = transform
        self.target_transform = target_transform
        
        self.data=[]
        self.targets=[]
        
        self.data= [(self.loader(s[0])) for s in self.samples]
        self.targets= [(s[1]) for s in self.samples]
        
        self.semi_targets = torch.zeros(len(self.targets), dtype=torch.int64)

        # self.alpha_uniform = Uniform(0.5, 1.5)
        # self.beta_uniform = Uniform(0.5, 2)

        # self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        # self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
    
    def __getitem__(self, index):
        """Override the original method of the CIFAR10 class.
        Args:
            index (int): Index
        Returns:
            triple: (image, target, index) where target is index of the target class.
        """
        # if self.train:
        #     img, target = self.data[index], self.targets[index]
        
        img, target, semi_target = self.data[index], self.targets[index], int(self.semi_targets[index])
        
        """
        transform_0 = transforms.Compose([ # transforms.RandomCrop((450,450)),
                                transforms.Resize((32,32)),
                                # transforms.Resize((96,96)),
                                transforms.ToTensor()
                                ])
        transform_1 = transforms.Compose([ # transforms.Resize((256, 256)),
                                transforms.Resize((32,32)),
                                # transforms.RandomCrop((32,32)),
                                transforms.ToTensor()
                                ])
        if target==0:
            img=transform_0(img)
        else:
            img=transform_1(img)
        """
        

        # to add GG noise
        """
        if(self.train):
            alpha_sample = self.alpha_uniform.sample(sample_shape=(3, 32, 32))
            beta_sample = self.beta_uniform.sample(sample_shape=(3, 32, 32))
            gg_sample = sample_ggd(torch.zeros_like(alpha_sample), alpha_sample, beta_sample)
            gg_sample /= 32.0
            img += gg_sample
        """

        # to add salt and pepper noise
        """
        if(self.train):
            img = np.array(img)
            # print("Type of image is:", type(img))            
            s_vs_p = 0.5
            amount = 0.10
            out = np.copy(img)
            # Salt mode
            # print("The size of image is:", img.size)
            num_salt = np.ceil(amount * img.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
            for i in img.shape]
            out[tuple(coords)] = 1
            # Pepper mode
            num_pepper = np.ceil(amount* img.size * (1. - s_vs_p))
            # print("The shape of image is:", img.shape)
            coords = [np.random.randint(0, i - 1, int(num_pepper))
            for i in img.shape]
            out[tuple(coords)] = 0
            img = Image.fromarray(out)
        """
        img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        # print(type(img))

        return img, target, semi_target, index
        # return img, target_, index, target  # only line changed

    def __len__(self):
        return len(self.samples)