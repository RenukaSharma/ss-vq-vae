import click
import torch
import logging
import random
import numpy as np
from networks.main import build_autoencoder
import sys
from PIL import Image
import os
import cv2
from tqdm import tqdm
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time

# model_path = os.path.join(_RESULTS_DIR, 'model.tar')

model_path = os.path.join(_RESULTS_DIR, 'model.tar')
# map_location = 'cpu'
map_location = 'cuda:0'
model_dict = torch.load(model_path, map_location=map_location)

print(type(model_dict))
print(model_dict.keys())
print(type(model_dict['ae_net_dict']))

ae_net = None
net_name = 'cifar10_LeNet'
if ae_net is None:
    ae_net = build_autoencoder(net_name)
ae_net.load_state_dict(model_dict['ae_net_dict'])
ae_net = ae_net.to(device=map_location)

# print(ae_net)

# Defining input
image_folder = _DATA_DIR#/0004_img.png'
gt_folder = _DATA_DIR#/0004_seg_abnormal.png'

# condition whether patches should be extracted
# no_patch = not class_name in ['wood', 'grid', 'leather', 'tile', 'carpet']
no_patch = False
step = 1
patch_size = 170
down_sample = 1 #5, to get the 34x43 image

transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
# iterating over each image
for file_name in tqdm(os.listdir(image_folder)):    
    if not file_name.endswith('_seg_abnormal.png'):

        file_name = '0009_img.png'
        
        normal_data = []
        abnormal_data = []
        normal_images = 0
        anomalous_images = 0

        # Reading images
        image_image = Image.open(os.path.join(image_folder, file_name)).convert('RGB') #.resize((32,32))
        image = np.asarray(image_image)
        print("Opened the image", file_name)

        # reading the mask
        mask_name = file_name[:4] + '_seg_abnormal.png' 
        mask = np.asarray(Image.open(os.path.join(gt_folder, mask_name)).convert('L'))
        print("Extracted the mask", mask_name)

        mat = np.zeros((image.shape[0], image.shape[1]))
        print("Size of matrix:", mat.shape)
        count = 0
        loopStartTime = time.time()
        

        # running the model on a patch
        input = transform(image)
        # print("Shape of input",input.shape)
       
        input = torch.unsqueeze(input, 0)
        # print("Shape of input",input.shape)
        input = input.to(device = map_location)
        outputs_encoded_mu, outputs_encoded_beta, outputs_recons_mu, outputs_recons_beta, sample = ae_net(input, ablation_type='A', eps=1e-5)
        
        # scores = torch.log(torch.norm(sample, dim=1) ** 2)
        # scores = (torch.norm(sample, dim=1) ** 2)
        scores = outputs_recons_mu ** 2
        scores = torch.squeeze(scores)
        
        mat = scores.cpu().detach().numpy()
        mat = mat.mean(axis=0)
        print("mat.shape",mat.shape)        

        # if(scores >= 234.858):
        #     mat[x_start + int(patch_size/2.0), y_start + int(patch_size/2.0)] = 1
        # else:
        #     mat[x_start + int(patch_size/2.0), y_start + int(patch_size/2.0)] = 0

        # mat[x_start + int(patch_size/2.0), y_start + int(patch_size/2.0)] = scores
        
        
        print("Matrix:\n",mat)
        np.save(_RESULTS_DIR+file_name[:4]+"_mat", mat)
        print("max val:", mat.max())
        print("min val:", mat.min())
        print("mean val:", mat.mean())
        mat = (255.0 *(mat - mat.min())/(mat.max() - mat.min())).astype('uint8')
        
        # plt.imshow(mat, interpolation='nearest')

        # imgplot = plt.imshow(mat, interpolation="nearest")
        # plt.axis('off')
        plt.imsave(_RESULTS_DIR+file_name[:4]+"_heatmap.png", mat)
        # plt.colorbar()
        # default colormap for plt is virdis
        plt.close()

    sys.exit()

# outputs_encoded_mu, outputs_encoded_beta, outputs_recons_mu, outputs_recons_beta, sample = ae_net(input, ablation_type='A', eps=1e-5)