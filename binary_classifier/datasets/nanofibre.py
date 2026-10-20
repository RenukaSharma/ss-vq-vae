from torch.utils.data import Subset, random_split, ConcatDataset
import PIL
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

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Nanofibre_Dataset(TorchvisionDataset):

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

        # MNIST preprocessing: feature scaling to [0, 1]
        transform = transforms.ToTensor()
        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))
        # idx, _, semi_targets = create_semisupervised_setting(np.asarray(train_set.targets), self.normal_classes,
        #                                                      self.outlier_classes, self.known_outlier_classes,
        #                                                      ratio_known_normal, ratio_known_outlier, ratio_pollution)
        # assert len(idx) == len(semi_targets)
        train_set = MyNanofibreDataset(root=self.root + '/train', train=True, transform=transform, download=False,
                                     target_transform=target_transform)
        print("#####Length of full train_set is", len(train_set))
        # print(train_set.targets, dir(train_set))
        print(max(train_set.targets), sum(train_set.targets))
        idx, _, semi_targets = create_semisupervised_setting(np.asarray(train_set.targets), self.normal_classes,
                                                             self.outlier_classes, self.known_outlier_classes,
                                                             ratio_known_normal, ratio_known_outlier, ratio_pollution)
        train_set.semi_targets[idx] = torch.tensor(semi_targets)  # set respective semi-supervised labels

        self.train_set = Subset(train_set, idx)
        logger.info("The length of train set is %d" % len(self.train_set))
        print("The length of self.train_set is", len(self.train_set))
        self.test_set = MyNanofibreDataset(root=self.root + '/test', train=False, transform=transform, download=False,
                                         target_transform=target_transform)
        logger.info("The length of test set is %d" % len(self.test_set))
        print("The length of self.test_set is", len(self.test_set))


class MyNanofibreDataset(ImageFolder):  # MyCIFAR10(CIFAR10):
    """Torchvision CIFAR10 class with patch of __getitem__ method to also return the index of a data sample."""

    def __init__(self, root, junk=False, train=False, transform=None, target_transform=None, download=False):
        super(MyNanofibreDataset, self).__init__(root, transform=transform, target_transform=target_transform)
        # list of (path, class_to_idx[target]) = samples
        self.junk = junk
        self.train = train  # training set or test set
        if self.train:
            self.train_data = []
            self.train_labels = []
            logger = logging.getLogger()

            self.train_data = [(self.loader(s[0])) for s in self.samples]  # s[0] is the image path
            print("train_data length", len(self.train_data))
            print("train_data[0] shape", self.train_data[0].size)
            logger.info("train_data shape %s" % (self.train_data[0].size,))
            logger.info(("train_data length %d" % len(self.train_data)))
            # self.train_data= self.train_data.transpose(0, 2, 3, 1) # convert to HWC, not needed
            self.train_labels = [(s[1]) for s in self.samples]  # s[1] is the target, ie class_index of the target class
        else:

            self.test_data = [(self.loader(s[0])) for s in self.samples]
            # self.test_data= self.test_data.transpose(0, 2, 3, 1) # convert to HWC, not needed
            self.test_labels = [(s[1]) for s in self.samples]

        self.semi_targets = torch.zeros_like(torch.tensor(self.targets))

    def __getitem__(self, index):
        """Override the original method of the CIFAR10 class.
        Args:
            index (int): Index
        Returns:
            triple: (image, target, index) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
            # print("imggg: ", img.shape)
            
        else:
            img, target = self.test_data[index], self.test_labels[index]

        if self.junk:

            img = np.array(img)
            img = np.std(img) * np.random.standard_normal(img.shape) + np.mean(img)
            # print(type(img))
            img = Image.fromarray(img.astype(np.uint8))
        else:

            pass
        
        img = PIL.ImageOps.grayscale(img)
        # img.save("tryyy.png")

        # print(type(img))
        semi_target = int(self.semi_targets[index])

        transform_0 = transforms.Compose([
            # transforms.RandomCrop(256),
            # transforms.ToPILImage(),
            transforms.Resize((32, 32)), #changed for DCAE
            transforms.ToTensor()])
        # , transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1'))])
        transform_1 = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            # transforms.Resize((89,89)), #changed for DCAE
            transforms.ToTensor()])
        # # , transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1'))])
        # if target == 0:
        #     img = transform_0(img)
        # else:
        #     img = transform_1(img)
        if self.train:
            img = transform_0(img)
        else:
            img = transform_1(img)
        
        # print("imggg: ", img.shape)

        if self.target_transform is not None:
            target_ = self.target_transform(target)

        # if self.train and target_ > 0:
        #     print("Garbad")
        # print(torch.max(img), torch.min(img))
        # print(type(img))
        # print(img.shape, target_)
        return img, target_, semi_target, index  # only line changed

    def __len__(self):
        return len(self.samples)
