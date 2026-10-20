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
_DATA_DIR = os.environ.get('DATA_DIR', '.')
_RESULTS_DIR = os.environ.get('RESULTS_DIR', os.path.join('.', 'results'))
import copy
from PIL import ImageFile
import torch
import logging

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from torch.distributions.gamma import Gamma
from torch.distributions.uniform import Uniform

def sample_ggd(mu, alpha, beta):
    gamma = Gamma(1/beta, 1)
    y = gamma.rsample()
    p = torch.ones(mu.shape)*0.5
    s = torch.bernoulli(p) - 0.5
    return mu + 2*alpha*s*(y**(1/beta))

ImageFile.LOAD_TRUNCATED_IMAGES=True

class Corrosion_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class: int = 0, known_outlier_class: int = 1, n_known_outlier_classes: int = 1,
                 ratio_known_normal: float = 0.0, ratio_known_outlier: float = 0.0, ratio_pollution: float = 0.0, length: int=5000, category='carpet'):
        super().__init__(root)

        logger = logging.getLogger()

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

        # transform = transforms.Compose([ # transforms.Resize((256, 256)),
        #                         transforms.Resize((32,32)),
        #                         # transforms.RandomCrop((32,32)),
        #                         transforms.ToTensor()
        #                         ])
        
        # transform_2 = transforms.Compose([ # transforms.Resize((256, 256)),
        #                         transforms.RandomApply([ transforms.RandomAffine(degrees= 30)],p=1),
        #                         transforms.Resize((32,32)),
        #                         # transforms.RandomCrop((32,32)),
        #                         transforms.ToTensor()
        #                         ])
        # transform_3 = transforms.Compose([ # transforms.Resize((256, 256)),
        #                         transforms.RandomApply([ transforms.RandomAffine(degrees= 45)],p=1),
        #                         transforms.Resize((32,32)),
        #                         # transforms.RandomCrop((32,32)),
        #                         transforms.ToTensor()
        #                         ])
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        ####### Getting the patches ########

        load_data = True
        if load_data:
            # data_full = torch.load(os.path.join(_DATA_DIR, 'train_data_full.pt'))
            # label_full = torch.load(os.path.join(_DATA_DIR, 'train_label_full.pt'))

            data_full = torch.load(os.path.join(_DATA_DIR, 'train_data_10k.pt')) 
            label_full = torch.load(os.path.join(_DATA_DIR, 'train_label_10k.pt'))

            # data_label_full = zip(data_full, label_full)
            # data_label_full = list(data_label_full)
            # data_label_10k = random.sample(data_label_full, 16000)

            # data_full = []
            # label_full = []
            # for d, l in data_label_10k:
            #     data_full.append(d)
            #     label_full.append(l)

            # torch.save(data_full, os.path.join(_DATA_DIR, 'train_data_10k.pt')) 
            # torch.save(label_full, os.path.join(_DATA_DIR, 'train_label_10k.pt'))

            train_set = TensorToDataset(data_full, label_full, train=True)
            idx, _, semi_targets = create_semisupervised_setting(np.asarray(train_set.targets), self.normal_classes,
                                                                self.outlier_classes, self.known_outlier_classes,
                                                                ratio_known_normal, ratio_known_outlier, ratio_pollution)
            train_set.semi_targets[idx] = torch.tensor(semi_targets)  # set respective semi-supervised labels

            self.train_set = Subset(train_set, idx)
            logger.info("The length of train set is %d" % len(self.train_set))
            print("The length of self.train_set is", len(self.train_set))

            
            
        else:            

            image_folder = _DATA_DIR
            gt_folder = _DATA_DIR

            # condition whether patches should be extracted
            # no_patch = not class_name in ['wood', 'grid', 'leather', 'tile', 'carpet']
            no_patch = False
            step = 5
            patch_size = 128
            down_sample = 1 #5, to get the 34x43 image

            # initializing list for storing images
            
            anomaly_label = 1
            data = []
            label = []
            normal_data_full = []
            abnormal_data_full = []
            # normal_images = 0
            # anomalous_images = 0

            # iterating over each image
            for file_name in tqdm(os.listdir(image_folder)):

                # breaking the loop if sufficient examples are extracted
                # if limit_images is not None and normal_images + anomalous_images > limit_images:
                #     break
               
                normal_data = []
                abnormal_data = []
                normal_images = 0
                anomalous_images = 0

                # Reading images
                image = np.asarray(Image.open(os.path.join(image_folder, file_name)).convert('RGB'))
                print("Opened the image", file_name)

                # reading the mask
                mask_name = 'Mask' + file_name[5:] 
                # mask = np.asarray(Image.open(os.path.join(gt_folder, mask_name)).convert('L'))
                # mask = (Image.open(os.path.join(gt_folder, mask_name)))
                mask = np.round((cv2.imread(os.path.join(gt_folder, mask_name)))/255.0)
                print("Extracted the mask", mask_name)
                print("The size of image:", image.shape)
                print("The size of mask:", mask.shape)

                if not no_patch:

                    for x_start in range(0, image.shape[0], step):
                        for y_start in range(0, image.shape[1], step):

                            # checking if the shape is valid
                            if x_start + patch_size <= image.shape[0] and y_start + patch_size <= image.shape[1]:

                                # checking whether the patch is in foreground
                                patch = np.copy(image[x_start: x_start + patch_size, y_start: y_start + patch_size])

                                # computing the fraction of anomalous pixels
                                mask_patch = np.copy(mask[x_start: x_start + patch_size, y_start: y_start + patch_size, 0])
                                anomaly_fraction = np.sum(mask_patch) / (mask_patch.shape[0] * mask_patch.shape[1])
                                # if (anomaly_fraction / 255.0) > 0.5:
                                if (anomaly_fraction ) > 0.5:
                                    anomalous_images += 1
                                    # data.append(transform(patch[::down_sample, ::down_sample]))
                                    abnormal_data.append(transform(patch[::down_sample, ::down_sample]))
                                    # label.append(anomaly_label)
                                    print(anomaly_fraction)
                                    # plt.imshow(patch[::down_sample, ::down_sample])
                                    # plt.show()
                                elif anomaly_fraction == 0 :
                                    normal_images += 1
                                    # data.append(transform(patch[::down_sample, ::down_sample]))
                                    normal_data.append(transform(patch[::down_sample, ::down_sample]))
                                    # label.append(0)

                else:
                    patch = np.copy(image)
                    patch = img_as_ubyte(resize(patch, (patch_size, patch_size), anti_aliasing=True))
                    data.append(transform(patch[::down_sample, ::down_sample]))
                    if image_type != 'good':
                        current_label = anomaly_label
                        anomalous_images += 1
                    else:
                        current_label = class_id
                        normal_images += 1
                    label.append(current_label)
                    
                
                print("normal count:", len(normal_data))
                print("abnormal count:", len(abnormal_data))

                if(len(abnormal_data)==0):
                    normal_data = []
                elif(len(normal_data)>len(abnormal_data)):
                    length_ = len(abnormal_data)
                    
                    normal_data = random.sample(normal_data, length_)

                    print("type(normal_data):", type(normal_data))
                    print("type(abnormal_data):", type(abnormal_data))

                    for data in normal_data:
                        normal_data_full.append(data)
                    for data in abnormal_data:
                        abnormal_data_full.append(data)

                    print("After leveling, normal count:", len(normal_data_full))
                    print("After leveling, abnormal count:", len(abnormal_data_full))
                else:
                    pass
                               

            ####### End of getting the patches #######

            # creating dataloader object
            print('Anomalous images: {}, Normal Images: {}, percentile: {}'.format(len(abnormal_data_full), len(normal_data_full), len(normal_data_full) * 100 / (len(abnormal_data_full) + len(normal_data_full))))
            transform = transforms.ToTensor()

            data_full = []
            label_full = []
            for point in normal_data_full:
                data_full.append(point)
                label_full.append(0)
            for point in abnormal_data_full:
                data_full.append(point)
                label_full.append(1)

            torch.save(data_full, os.path.join(_DATA_DIR, 'train_data_full.pt')) 
            torch.save(label_full, os.path.join(_DATA_DIR, 'train_label_full.pt'))
            train_set = TensorToDataset(data_full, label_full, train=True)
            idx, _, semi_targets = create_semisupervised_setting(np.asarray(train_set.targets), self.normal_classes,
                                                                self.outlier_classes, self.known_outlier_classes,
                                                                ratio_known_normal, ratio_known_outlier, ratio_pollution)
            train_set.semi_targets[idx] = torch.tensor(semi_targets)  # set respective semi-supervised labels

            self.train_set = Subset(train_set, idx)
            logger.info("The length of train set is %d" % len(self.train_set))
            print("The length of self.train_set is", len(self.train_set))
            
            # data_loader = DataLoader(dset, batch_size=64, shuffle=True, num_workers=0)
        
        ##########################################
        #TEST SET#
        #Getting the patches
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        ####### Getting the patches ########
        load_data = True
        if load_data:
            data_full = torch.load(os.path.join(_DATA_DIR, 'test_data_2p5k.pt')) 
            label_full = torch.load(os.path.join(_DATA_DIR, 'test_label_2p5k.pt'))
            # data_label_full = zip(data_full, label_full)
            # data_label_full = list(data_label_full)
            # data_label_2_5k = random.sample(data_label_full, 2500)

            # data_full = []
            # label_full = []
            # for d, l in data_label_2_5k:
            #     data_full.append(d)
            #     label_full.append(l)

            # torch.save(data_full, os.path.join(_DATA_DIR, 'test_data_2p5k.pt')) 
            # torch.save(label_full, os.path.join(_DATA_DIR, 'test_label_2p5k.pt'))
            
            self.test_set = TensorToDataset(data_full, label_full, train=False)
            logger.info("The length of test set is %d" % len(self.test_set))
            print("The length of self.test_set is", len(self.test_set))
        
        else:
            image_folder = _DATA_DIR
            gt_folder = _DATA_DIR

            # condition whether patches should be extracted
            # no_patch = not class_name in ['wood', 'grid', 'leather', 'tile', 'carpet']
            no_patch = False
            step = 5
            patch_size = 128
            down_sample = 1 #5, to get the 34x43 image

            # initializing list for storing images
            
            anomaly_label = 1
            data = []
            label = []
            normal_data_full = []
            abnormal_data_full = []
            # normal_images = 0
            # anomalous_images = 0

            # iterating over each image
            for file_name in tqdm(os.listdir(image_folder)):

                # breaking the loop if sufficient examples are extracted
                # if limit_images is not None and normal_images + anomalous_images > limit_images:
                #     break
                
                normal_data = []
                abnormal_data = []
                normal_images = 0
                anomalous_images = 0

                # Reading images
                image = np.asarray(Image.open(os.path.join(image_folder, file_name)).convert('RGB'))
                print("Opened the image", file_name)

                # reading the mask
                mask_name = 'Mask' + file_name[5:] 
                # mask = np.asarray(Image.open(os.path.join(gt_folder, mask_name)).convert('L'))
                # mask = (Image.open(os.path.join(gt_folder, mask_name)))
                if os.path.exists(os.path.join(gt_folder, mask_name)):
                    mask = np.round((cv2.imread(os.path.join(gt_folder, mask_name)))/255.0)                    
                    print("Extracted the mask", mask_name)

                    if not no_patch:

                        for x_start in range(0, image.shape[0], step):
                            for y_start in range(0, image.shape[1], step):

                                # checking if the shape is valid
                                if x_start + patch_size <= image.shape[0] and y_start + patch_size <= image.shape[1]:

                                    # checking whether the patch is in foreground
                                    patch = np.copy(image[x_start: x_start + patch_size, y_start: y_start + patch_size])

                                    # computing the fraction of anomalous pixels
                                    mask_patch = np.copy(mask[x_start: x_start + patch_size, y_start: y_start + patch_size, 0])
                                    anomaly_fraction = np.sum(mask_patch) / (mask_patch.shape[0] * mask_patch.shape[1])
                                    # if (anomaly_fraction / 255.0) > 0.5:
                                    if (anomaly_fraction ) > 0.5:
                                        anomalous_images += 1
                                        # data.append(transform(patch[::down_sample, ::down_sample]))
                                        abnormal_data.append(transform(patch[::down_sample, ::down_sample]))
                                        # label.append(anomaly_label)
                                        print(anomaly_fraction)
                                        # plt.imshow(patch[::down_sample, ::down_sample])
                                        # plt.show()
                                    elif anomaly_fraction == 0 :
                                        normal_images += 1
                                        # data.append(transform(patch[::down_sample, ::down_sample]))
                                        normal_data.append(transform(patch[::down_sample, ::down_sample]))
                                        # label.append(0)

                    else:
                        patch = np.copy(image)
                        patch = img_as_ubyte(resize(patch, (patch_size, patch_size), anti_aliasing=True))
                        data.append(transform(patch[::down_sample, ::down_sample]))
                        if image_type != 'good':
                            current_label = anomaly_label
                            anomalous_images += 1
                        else:
                            current_label = class_id
                            normal_images += 1
                        label.append(current_label)
                        
                    
                    print("normal count:", len(normal_data))
                    print("abnormal count:", len(abnormal_data))

                    if(len(abnormal_data)==0):
                        normal_data = []
                    elif (len(normal_data)>len(abnormal_data)):
                        length_ = len(abnormal_data)
                        
                        normal_data = random.sample(normal_data, length_)

                        print("type(normal_data):", type(normal_data))
                        print("type(abnormal_data):", type(abnormal_data))

                        for data in normal_data:
                            normal_data_full.append(data)
                        for data in abnormal_data:
                            abnormal_data_full.append(data)

                        print("After leveling, normal count:", len(normal_data_full))
                        print("After leveling, abnormal count:", len(abnormal_data_full))
                    else:
                        pass
                
                else:
                    pass
            ####### End of getting the patches #######

            # creating dataloader object
            print('Anomalous images: {}, Normal Images: {}, percentile: {}'.format(len(abnormal_data_full), len(normal_data_full), len(normal_data_full) * 100 / (len(abnormal_data_full) + len(normal_data_full))))
            transform = transforms.ToTensor()

            data_full = []
            label_full = []
            for point in normal_data_full:
                data_full.append(point)
                label_full.append(0)
            for point in abnormal_data_full:
                data_full.append(point)
                label_full.append(1)

            torch.save(data_full, os.path.join(_DATA_DIR, 'test_data_full.pt')) 
            torch.save(label_full, os.path.join(_DATA_DIR, 'test_label_full.pt'))
            
            self.test_set = TensorToDataset(data_full, label_full, train=False)
            logger.info("The length of test set is %d" % len(self.test_set))
            print("The length of self.test_set is", len(self.test_set))
            
            

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
        
        # train_set = MyMVTec(root=self.root+'/train', train=True, transform=transform, target_transform=target_transform,
        #                       download=True)
        # train_set_2 = MyMVTec(root=self.root+'/train', train=True, transform=transform_2, target_transform=target_transform,
        #                       download=True)
        # train_set_3 = MyMVTec(root=self.root+'/train', train=True, transform=transform_3, target_transform=target_transform,
        #                       download=True)
        # print("The length of train_set is",len(train_set))
        # print("The length of train_set_2 is",len(train_set_2))
        # print("The length of train_set_3 is",len(train_set_3))

        # # Create semi-supervised setting
        # datasets =[]
        # idx, _, semi_targets = create_semisupervised_setting(np.array(train_set.targets), self.normal_classes,
        #                                                      self.outlier_classes, self.known_outlier_classes,
        #                                                      ratio_known_normal, ratio_known_outlier, ratio_pollution)
        # train_set.semi_targets[idx] = torch.tensor(semi_targets)[:train_set.semi_targets[idx].shape[0]]  # set respective semi-supervised labels
        # datasets.append(Subset(train_set, idx))

        # idx, _, semi_targets = create_semisupervised_setting(np.array(train_set_2.targets), self.normal_classes,
        #                                                      self.outlier_classes, self.known_outlier_classes,
        #                                                      ratio_known_normal, ratio_known_outlier, ratio_pollution)
        # train_set_2.semi_targets[idx] = torch.tensor(semi_targets)[:train_set_2.semi_targets[idx].shape[0]]  # set respective semi-supervised labels
        # datasets.append(Subset(train_set_2, idx))

        # idx, _, semi_targets = create_semisupervised_setting(np.array(train_set_3.targets), self.normal_classes,
        #                                                      self.outlier_classes, self.known_outlier_classes,
        #                                                      ratio_known_normal, ratio_known_outlier, ratio_pollution)
        # train_set_3.semi_targets[idx] = torch.tensor(semi_targets)[:train_set_3.semi_targets[idx].shape[0]]  # set respective semi-supervised labels
        # datasets.append(Subset(train_set_3, idx))

        # # Subset train_set to semi-supervised setup
        # self.train_set = ConcatDataset (datasets)
        
        # logger = logging.getLogger()
        # logger.info('The length of train_set is %d' % len(self.train_set))

        
                    
        # self.test_set = ConcatDataset   (
        #                                 [MyMVTec(root=self.root+'/test', train=False, transform=transform, 
        #                                 target_transform=target_transform, download=True),
        #                                 MyMVTec(root=self.root+'/test', train=False, transform=transform_2, 
        #                                 target_transform=target_transform, download=True),
        #                                 MyMVTec(root=self.root+'/test', train=False, transform=transform_3, 
        #                                 target_transform=target_transform, download=True)
        #                                 ]
        #                                 )
        ########################

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

    def __len__(self):
        return len(self.tensor_batch)

    def __getitem__(self, index):
        img, target, semi_target = self.tensor_batch[index], self.targets[index], int(self.semi_targets[index])
        # img = Image.fromarray(img)
        return img, target, semi_target, index
        # return img, target, semi_target, index

class MyCorrosion_Dataset(ImageFolder):#MyCIFAR10(CIFAR10):
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

        # to salt and pepper noise
    
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
        img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        # print(type(img))

        return img, target, semi_target, index
        # return img, target_, index, target  # only line changed

    def __len__(self):
        return len(self.samples)