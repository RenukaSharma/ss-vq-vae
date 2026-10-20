from torch.utils.data import Subset, random_split, ConcatDataset
from PIL import Image
from torch import randperm
from torchvision.datasets import CIFAR10
from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import create_semisupervised_setting
# from .preprocessing import get_target_label_idx, global_contrast_normalization
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data as data
from torchvision.datasets import ImageFolder
import os
import copy
from PIL import ImageFile
import logging
import torch
import random

from datetime import datetime


from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from torch.distributions.gamma import Gamma
from torch.distributions.uniform import Uniform

ImageFile.LOAD_TRUNCATED_IMAGES=True

def sample_ggd(mu, alpha, beta):
    gamma = Gamma(1/beta, 1)
    y = gamma.rsample()
    p = torch.ones(mu.shape)*0.5
    s = torch.bernoulli(p) - 0.5
    return mu + 2*alpha*s*(y**(1/beta))

class Malaria_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class=3, known_outlier_class=3, n_known_outlier_classes=0,
                 ratio_known_normal=0.0, ratio_known_outlier=0.0, ratio_pollution=0.0):
        # root: str, normal_class=3,impure_class=-1,impure_frac=0.0,junk=False,junk_percentage=0.0,
        # semi_sp_class=-1,semi_sp_impure_frac=0.0,semi_sp_junk=0.0,small_dataset=False
        super().__init__(root)
        logger = logging.getLogger()
        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 2))
        self.outlier_classes.remove(normal_class)
        self.outlier_classes = tuple(self.outlier_classes)

        if n_known_outlier_classes == 0:
            self.known_outlier_classes = ()
        elif n_known_outlier_classes == 1:
            self.known_outlier_classes = tuple([known_outlier_class])
        else:
            self.known_outlier_classes = tuple(random.sample(self.outlier_classes, n_known_outlier_classes))

        
        # transform = transforms.ToTensor()
        # target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

      
        # data_full = torch.load(os.path.join(_DATA_DIR, 'train_data_1209onwards_10k.pt'))
        # label_full = torch.load(os.path.join(_DATA_DIR, 'train_label_1209onwards_10k.pt'))

        data_full = torch.load(os.path.join(_DATA_DIR, 'train_data_20k.pt'))
        label_full = torch.load(os.path.join(_DATA_DIR, 'train_label_20k.pt'))        

        train_set = TensorToDataset(data_full, label_full, train=True)
        print("Length of train set before creating semi-supervised setting:", len(train_set))
        idx, _, semi_targets = create_semisupervised_setting(np.asarray(train_set.targets), self.normal_classes,
                                                            self.outlier_classes, self.known_outlier_classes,
                                                            ratio_known_normal, ratio_known_outlier, ratio_pollution)
        train_set.semi_targets[idx] = torch.tensor(semi_targets)  # set respective semi-supervised labels

        self.train_set = Subset(train_set, idx)
        logger.info("The length of train set is %d" % len(self.train_set))
        print("The length of self.train_set is", len(self.train_set))


        ##########################################
        #TEST SET#
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # data_full = torch.load(os.path.join(_RESULTS_DIR, 'test_data_2p5k.pt'))
        # label_full = torch.load(os.path.join(_RESULTS_DIR, 'test_label_2p5k.pt'))

        data_full = torch.load(os.path.join(_DATA_DIR, 'test_data_5k.pt'))
        label_full = torch.load(os.path.join(_DATA_DIR, 'test_label_5k.pt'))
        
        self.test_set = TensorToDataset(data_full, label_full, train=False)
        logger.info("The length of test set is %d" % len(self.test_set))
        print("$$$$$$$$$$$$$$$$The length of self.test_set is", len(self.test_set))

        

class TensorToDataset(Dataset):
    def __init__(self, tensor_batch, targets, train=False):
        self.train = train
        self.tensor_batch = tensor_batch
        self.targets = targets
        self.semi_targets = torch.zeros_like(torch.tensor(self.targets))
        # self.semi_targets = []
        # for index in tqdm(range(len(target))):
        #     self.semi_targets.append(-1.0 * target[index])
        # self.transform = transform
        self.length = len(tensor_batch)

        self.alpha_uniform = Uniform(0.5, 1.5)
        self.beta_uniform = Uniform(0.5, 2)


    def __len__(self):
        return len(self.tensor_batch)

    def __getitem__(self, index):
        img, target, semi_target = self.tensor_batch[index], self.targets[index], int(self.semi_targets[index])
        # img = Image.fromarray(img)
        
        # to add GG noise
        """
        if(self.train):
            alpha_sample = self.alpha_uniform.sample(sample_shape=(3, 32, 32))
            beta_sample = self.beta_uniform.sample(sample_shape=(3, 32, 32))
            gg_sample = sample_ggd(torch.zeros_like(alpha_sample), alpha_sample, beta_sample)
            gg_sample /= 32.0
            img += gg_sample
        """
        
        """
        # to add S and P noise
        if(self.train):
            img = np.array(img)
            # print("Type of image is:", type(img))            
            s_vs_p = 0.5
            amount = 0.40
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
            img = out
            # img = Image.fromarray(out)
            # print("^^^^^^^^^^^^^^^^^^^^^^^^^^Added SP noise")
        """
        return img, target, semi_target, index
        # return img, target, semi_target, index