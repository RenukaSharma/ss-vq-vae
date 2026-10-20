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
_DATA_DIR = os.environ.get('DATA_DIR', '.')
_RESULTS_DIR = os.environ.get('RESULTS_DIR', os.path.join('.', 'results'))
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

        ####### Getting the patches ########

        # load_data = False
        load_data = True
        if load_data:
            data_full = torch.load(os.path.join(_DATA_DIR, 'train_data_take-2.pt'))
            label_full = torch.load(os.path.join(_DATA_DIR, 'train_label_take-2.pt'))

            # data_full = torch.load(os.path.join(_DATA_DIR, 'train_data_full.pt'))
            # label_full = torch.load(os.path.join(_DATA_DIR, 'train_label_full.pt'))

            # data_label_full = zip(data_full, label_full)
            # data_label_full = list(data_label_full)
            # data_label_10k = random.sample(data_label_full, 15000)

            # data_full = []
            # label_full = []
            # for d, l in data_label_10k:
            #     data_full.append(d)
            #     label_full.append(l)

            # torch.save(data_full, _DATA_DIR+str(datetime.now().timestamp())+'.pt') 
            # torch.save(label_full, _DATA_DIR+str(datetime.now().timestamp())+'.pt')

            train_set = TensorToDataset(data_full, label_full, train=True)
            print("Length of train set before creating semi-supervised setting:", len(train_set))
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
            mask_all_folder = _DATA_DIR

            # condition whether patches should be extracted
            # no_patch = not class_name in ['wood', 'grid', 'leather', 'tile', 'carpet']
            no_patch = False
            step = 5
            patch_size = 170
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
                if int(file_name[:4]) in range(1,1209):
                    print("file index is ", int(file_name[:4]))
                    file_idx = int(file_name[:4])
                    
                    normal_data = []
                    abnormal_data = []
                    normal_images = 0
                    anomalous_images = 0

                    # Reading images
                    image = np.asarray(Image.open(os.path.join(image_folder, file_name)).convert('RGB'))
                    print("Opened the image", file_name)

                    mask_all_name = "{:04d}".format(file_idx) + "_seg_abnormal.png"
                    print("~~~~~~~~Extracting mask_all", mask_all_name)
                    mask_all = np.asarray(Image.open(os.path.join(mask_all_folder, mask_all_name)).convert('L'))
                    
                    for x_start in range(0, image.shape[0], step):
                        for y_start in range(0, image.shape[1], step):
                            # checking if the shape is valid
                            if x_start + patch_size <= image.shape[0] and y_start + patch_size <= image.shape[1]:
                                # checking whether the patch is in foreground
                                patch = np.copy(image[x_start: x_start + patch_size, y_start: y_start + patch_size])

                                # computing the fraction of anomalous pixels
                                mask_all_patch = np.copy(mask_all[x_start: x_start + patch_size, y_start: y_start + patch_size])
                                anomaly_fraction_all = np.sum(mask_all_patch) / (mask_all_patch.shape[0] * mask_all_patch.shape[1])

                                if (anomaly_fraction_all / 255.0) == 0:
                                    
                                    normal_images += 1
                                    # data.append(transform(patch[::down_sample, ::down_sample]))
                                    normal_data.append(transform(patch[::down_sample, ::down_sample]))
                                    # label.append(0)
                    
                    
                    # for class_ in ['difficult','gametocyte','leukocyte','ring', 'schizont', 'trophozoite']:
                    for class_ in ['ring', 'schizont', 'trophozoite']: # for take 2
                    # for class_ in ['gametocyte','ring', 'schizont', 'trophozoite']: # for tensors 1
                    # for class_ in [ring', 'schizont', 'trophozoite']: # for tensors
                        
                        # reading the mask                        
                        mask_name = "{:04d}".format(file_idx-1) + "_seg_abnormal_"+class_+".png"                        
                        mask = np.asarray(Image.open(os.path.join(gt_folder, mask_name)).convert('L'))
                        
                        print("Extracted the ",class_,"mask", mask_name)

                        for x_start in range(0, image.shape[0], step):
                            for y_start in range(0, image.shape[1], step):

                                # checking if the shape is valid
                                if x_start + patch_size <= image.shape[0] and y_start + patch_size <= image.shape[1]:

                                    # checking whether the patch is in foreground
                                    patch = np.copy(image[x_start: x_start + patch_size, y_start: y_start + patch_size])

                                    # computing the fraction of anomalous pixels
                                    mask_patch = np.copy(mask[x_start: x_start + patch_size, y_start: y_start + patch_size])
                                    anomaly_fraction = np.sum(mask_patch) / (mask_patch.shape[0] * mask_patch.shape[1])

                                    if (anomaly_fraction / 255.0) > 0.5:
                                        anomalous_images += 1
                                        # data.append(transform(patch[::down_sample, ::down_sample]))
                                        abnormal_data.append(transform(patch[::down_sample, ::down_sample]))
                                        # label.append(anomaly_label)
                                        # print(anomaly_fraction)
                                        # plt.imshow(patch[::down_sample, ::down_sample])
                                        # plt.show()
                    print("normal count :", len(normal_data))
                    print("abnormal count :", len(abnormal_data))
                    
                    if(len(abnormal_data)==0):
                        normal_data = []
                    else:
                        length_ = len(abnormal_data)
                        
                        normal_data = random.sample(normal_data, length_)

                        print("type(normal_data):", type(normal_data))
                        print("type(abnormal_data):", type(abnormal_data))

                        for data in normal_data:
                            normal_data_full.append(data)
                        for data in abnormal_data:
                            abnormal_data_full.append(data)

                        print("After leveling, normal count :", len(normal_data_full))
                        print("After leveling, abnormal count :", len(abnormal_data_full))

                        if len(abnormal_data_full) >= 10000:
                            break
                    
                    if len(abnormal_data_full) >= 10000:
                        break
                    
                # print("len(data[label==0]):",len(data[label==0]))
                # print("len of data:", len(data))
                # print("len of label:", len(label))
                # print("normal count:", normal_images)
                # print("abnormal count:", anomalous_images)

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

            torch.save(data_full, os.path.join(_DATA_DIR, 'train_data_take-2.pt')) 
            torch.save(label_full, os.path.join(_DATA_DIR, 'train_label_take-2.pt'))
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
        # load_data = False
        if load_data:
            data_full = torch.load(os.path.join(_DATA_DIR, 'test_data.pt'))
            label_full = torch.load(os.path.join(_DATA_DIR, 'test_label.pt'))

            # data_full = torch.load(os.path.join(_DATA_DIR, 'test_data_2p5k.pt')) 
            # label_full = torch.load(os.path.join(_DATA_DIR, 'test_label_2p5k.pt'))
            # data_label_full = zip(data_full, label_full)
            # data_label_full = list(data_label_full)
            # data_label_2_5k = random.sample(data_label_full, 2500)

            # data_full = []
            # label_full = []
            # for d, l in data_label_2_5k:
            #     data_full.append(d)
            #     label_full.append(l)
            
            # torch.save(data_full, _DATA_DIR+str(datetime.now().timestamp())+'.pt') 
            # torch.save(label_full, _DATA_DIR+str(datetime.now().timestamp())+'.pt')

            self.test_set = TensorToDataset(data_full, label_full, train=False)
            logger.info("The length of test set is %d" % len(self.test_set))
            print("$$$$$$$$$$$$$$$$The length of self.test_set is", len(self.test_set))

        else:

            image_folder = _DATA_DIR # starts from 1
            gt_folder = _DATA_DIR # starts from 0
            mask_all_folder = _DATA_DIR # starts from 1

            # condition whether patches should be extracted
            # no_patch = not class_name in ['wood', 'grid', 'leather', 'tile', 'carpet']
            no_patch = False
            step = 5
            patch_size = 170
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
                count_dict = {}
                    
                normal_data = []
                abnormal_data = []
                normal_images = 0
                anomalous_images = 0

                # Reading images
                image = np.asarray(Image.open(os.path.join(image_folder, file_name)).convert('RGB'))
                print("Opened the image", file_name)

                # reading the mask
                if int(file_name[:4]) in range(1209, 1329):
                    print("file index is ", int(file_name[:4]))
                    file_idx = int(file_name[:4])
                    
                    mask_all_name = "{:04d}".format(file_idx) + "_seg_abnormal.png"                    
                    mask_all = np.asarray(Image.open(os.path.join(mask_all_folder, mask_all_name)).convert('L'))
                    print("Extracted the mask", mask_all_name)

                    for x_start in range(0, image.shape[0], step):
                        for y_start in range(0, image.shape[1], step):

                            # checking if the shape is valid
                            if x_start + patch_size <= image.shape[0] and y_start + patch_size <= image.shape[1]:

                                # checking whether the patch is in foreground
                                patch = np.copy(image[x_start: x_start + patch_size, y_start: y_start + patch_size])

                                # computing the fraction of anomalous pixels

                                mask_all_patch = np.copy(mask_all[x_start: x_start + patch_size, y_start: y_start + patch_size])
                                anomaly_fraction_all = np.sum(mask_all_patch) / (mask_all_patch.shape[0] * mask_all_patch.shape[1])

                                if (anomaly_fraction_all / 255.0) > 0.5:
                                    anomalous_images += 1
                                    # data.append(transform(patch[::down_sample, ::down_sample]))
                                    abnormal_data.append(transform(patch[::down_sample, ::down_sample]))
                                    # label.append(anomaly_label)
                                    # print(anomaly_fraction)
                                    # plt.imshow(patch[::down_sample, ::down_sample])
                                    # plt.show()
                                elif (anomaly_fraction_all / 255.0) == 0.0 :
                                    normal_images += 1
                                    # data.append(transform(patch[::down_sample, ::down_sample]))
                                    normal_data.append(transform(patch[::down_sample, ::down_sample]))
                                    # label.append(0)
                        
                    print("normal count :", len(normal_data))
                    print("abnormal count :", len(abnormal_data))

                    if(len(abnormal_data)==0):
                        normal_data = []
                    else:
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

                        if len(abnormal_data_full)>=2500:
                            break
                            
                    if len(abnormal_data_full)>=2500:
                        break      
                        # print("len(data[label==0]):",len(data[label==0]))
                        # print("len of data:", len(data))
                        # print("len of label:", len(label))
                        # print("normal count:", normal_images)
                        # print("abnormal count:", anomalous_images)

            ####### End of getting the patches #######

            # creating dataloader object
            # print('Anomalous images: {}, Normal Images: {}, percentile: {}'.format(len(abnormal_data_full), len(normal_data_full), len(normal_data_full) * 100 / (len(abnormal_data_full) + len(normal_data_full))))
            transform = transforms.ToTensor()

            data_full = []
            label_full = []
            for point in normal_data_full:
                data_full.append(point)
                label_full.append(0)
            for point in abnormal_data_full:
                data_full.append(point)
                label_full.append(1)

            torch.save(data_full, os.path.join(_DATA_DIR, 'test_data.pt')) 
            torch.save(label_full, os.path.join(_DATA_DIR, 'test_label.pt'))
            
            self.test_set = TensorToDataset(data_full, label_full, train=False)
            logger.info("The length of test set is %d" % len(self.test_set))
            print("The length of self.test_set is", len(self.test_set))
            
            # data_loader = DataLoader(dset, batch_size=64, shuffle=True, num_workers=0)

            ##########################################
            """
            train_set = MyMalariaDataset(root=self.root + '/train', train=True, transform=transform, download=False,
                                        target_transform=target_transform)
            print("#####Length of full train_set is", len(train_set))

            idx, _, semi_targets = create_semisupervised_setting(np.asarray(train_set.targets), self.normal_classes,
                                                                self.outlier_classes, self.known_outlier_classes,
                                                                ratio_known_normal, ratio_known_outlier, ratio_pollution)
            train_set.semi_targets[idx] = torch.tensor(semi_targets)  # set respective semi-supervised labels

            self.train_set = Subset(train_set, idx)
            logger.info("The length of train set is %d" % len(self.train_set))
            print("The length of self.train_set is", len(self.train_set))
            self.test_set = MyMalariaDataset(root=self.root + '/test', train=False, transform=transform, download=False,
                                            target_transform=target_transform)
            logger.info("The length of test set is %d" % len(self.test_set))
            print("The length of self.test_set is", len(self.test_set))
            """



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
        """
        # to add GG noise
        if(self.train):
            alpha_sample = self.alpha_uniform.sample(sample_shape=(3, 32, 32))
            beta_sample = self.beta_uniform.sample(sample_shape=(3, 32, 32))
            gg_sample = sample_ggd(torch.zeros_like(alpha_sample), alpha_sample, beta_sample)
            gg_sample /= 32.0
            img += gg_sample
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
        
        return img, target, semi_target, index
        # return img, target, semi_target, index