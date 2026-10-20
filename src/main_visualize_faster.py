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
transform1 = transforms.ToTensor()
# iterating over each image
for file_name in tqdm(os.listdir(image_folder)):
    
    if not file_name.endswith('_seg_abnormal.png'):
        
        data = torch.zeros((256, 3, 32, 32))
        masks = []

        # Reading images
        image_image = Image.open(os.path.join(image_folder, file_name)).convert('RGB') #.resize((32,32))
        image = np.asarray(image_image)
        print("Opened the image", file_name)

        # reading the mask
        mask_name = file_name[:4] + '_seg_abnormal.png' 
        mask = np.asarray(Image.open(os.path.join(gt_folder, mask_name)).convert('L'))
        print("Extracted the mask", mask_name)
        
        count = 0
        start_time = time.time()
        scores_whole_img = torch.empty()
        for x_start in tqdm(range(0, image.shape[0], step)):
            for y_start in range(0, image.shape[1], step):

                # checking if the shape is valid
                if x_start + patch_size <= image.shape[0] and y_start + patch_size <= image.shape[1]:

                    # checking whether the patch is in foreground
                    patch = np.copy(image[x_start: x_start + patch_size, y_start: y_start + patch_size])
                    
                    input = transform(patch)
                    # data.append(input)
                    data[count] = input
                    count += 1
                    
                    if (count>=256):
                        data = data.to(device=map_location)
                        ae_net = ae_net.to(device=map_location)
                        # running the model on a batch
                        outputs_encoded_mu, outputs_encoded_beta, outputs_recons_mu, outputs_recons_beta, sample = ae_net(data, ablation_type='A', eps=1e-5)
                        scores = torch.log(torch.norm(sample, dim=1) ** 2)
                        scores_whole_img.cat(scores)
                        count = 0    
                        data = torch.zeros((256, 3, 32, 32))    
                 
                    
        # print(type(data))
        
        # torch.save(data, _RESULTS_DIR+file_name[:4]+'_data.pt')

        # data = np.array(data)
        # data = torch.tensor(data)
        # print("data.shape:", data.shape)
        # data = data.to(device=map_location)
        # ae_net = ae_net.to(device=map_location)
        # outputs_encoded_mu, outputs_encoded_beta, outputs_recons_mu, outputs_recons_beta, sample = ae_net(data, ablation_type='A', eps=1e-5)
        # scores = torch.log(torch.norm(sample, dim=1) ** 2)

        print("Duration:", time.time()-start_time)
        # mat = np.zeros((image.shape[0], image.shape[1]))
        # count = 0 
        # for x_start in range(0, image.shape[0], step):
        #     for y_start in range(0, image.shape[1], step):
        #         # checking if the shape is valid
        #         if x_start + patch_size <= image.shape[0] and y_start + patch_size <= image.shape[1]:
        #             mat[x_start + int(patch_size/2.0), y_start + int(patch_size/2.0)] = scores[count]
        #             count += 1


        # np.save(_RESULTS_DIR+file_name[:4]+"_logscores_mat", mat)
                
        # img_plot = Image.fromarray(data, 'RGB')
        # img_plot.save('my.png')
        # imgplot = plt.imshow(mat, interpolation="nearest")
        # plt.axis('off')
        # plt.savefig(_RESULTS_DIR+file_name[:4]+"_heatmap.png")
        # # default colormap for plt is virdis
        # plt.close()