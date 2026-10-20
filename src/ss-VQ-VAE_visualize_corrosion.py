import click
from cv2 import NORMAL_CLONE
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

from utils.config import Config
from utils.visualization.plot_images_grid import plot_images_grid, plot_matrix_grid
from DeepSAD import DeepSAD
from datasets.main import load_dataset
import torchvision.transforms as transforms
from sklearn.manifold import TSNE

from networks.main import build_network, build_autoencoder

model_path = os.path.join(_RESULTS_DIR, "model.tar")
# model_path = os.path.join(_RESULTS_DIR, "model.tar") # full sup, corr, 0.99 weight decay, wo dtaa variance, 15k epochs
# model_path = os.path.join(_RESULTS_DIR, "model.tar") # full sup ss-gVAE # optimal_threshold = 3.1197e+06

# model_path = os.path.join(_RESULTS_DIR, "model.tar") # 0 sup ss-gVAE # optimal_threshold = 329.87

eta = 1.0
recon_param = 1.0
# device = "cuda:0"
device = "cpu"

deepSAD = DeepSAD(eta, recon_param, xp_path="")
# net_name = "cifar10_LeNet"
# net_name = "mvtec_net"
net_name = "vq_vae_net"
num_hiddens = 128
num_residual_hiddens= 32
num_residual_layers= 2
embedding_dim= 64
num_embeddings= 512
commitment_cost= 1
build_autoencoder(net_name=net_name, num_hiddens= num_hiddens, num_residual_hiddens= num_residual_hiddens, num_residual_layers= num_residual_layers, embedding_dim= embedding_dim, num_embeddings= num_embeddings, commitment_cost= commitment_cost, weight_decay= 0.5e-6)

deepSAD.load_model(model_path=model_path, load_ae=True, map_location=device)

deepSAD.ae_net = deepSAD.ae_net.to(device)
deepSAD.ae_net.eval()

dataset_name = "mvtec"
category =  "carpet"
# data_path = _DATA_DIR+category+"-with-novel-ano-in-test"
normal_class = 0
known_outlier_class = 1
n_known_outlier_classes = 1
ratio_known_normal = 0
ratio_known_outlier = 0.49
ratio_pollution = 0
seed = 1 

image_folder = _DATA_DIR
gt_folder = _DATA_DIR
file_name = "Image28.JPG"

image_image = Image.open(os.path.join(image_folder, file_name)).convert('RGB') #.resize((32,32))
image = np.asarray(image_image)
print("Opened the image", file_name)

# reading the mask
mask_name = 'Mask28.png' 
mask = np.asarray(Image.open(os.path.join(gt_folder, mask_name)).convert('L'))
print("Extracted the mask", mask_name)

mat = np.zeros((image.shape[0], image.shape[1]))
print("Size of matrix:", mat.shape)
count = 0
loopStartTime = time.time()

transform = transforms.Compose([      
            transforms.ToPILImage(),      
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
# running the model on a patch
input = transform(image)
# print("Shape of input",input.shape)

input = torch.unsqueeze(input, 0)
# print("Shape of input",input.shape)
input = input.to(device = device)
eps = 1e-5

vq_loss, data_recon, z, quantized, perplexity = deepSAD.ae_net(input)
"""
reconImg, sampleRecon, samplingParasAll, out_class = deepSAD.ae_net(input, ablation_type='A', eps=eps)
# outputs_recons_mu, outputs_recons_beta = samplingParasAll['final']['mu'], samplingParasAll['final']['sigma']
outputs_recons_mu, outputs_recons_alpha, outputs_recons_beta = samplingParasAll['final']['mu'], samplingParasAll['final']['sigma'], samplingParasAll['final']['beta']

dist_first = 0.0
dist_second = 0.0
for key in samplingParasAll['hidden'].keys():       
    para = samplingParasAll['hidden'][key]
    sample = para['sample']
    outputs_encoded_mu = para['mu']
    outputs_encoded_alpha = para['sigma']
    outputs_encoded_beta = para['beta']
"""
print("Got the results after processing the data through the network")
print(outputs_encoded_mu.shape)
outputs_encoded_mu = np.squeeze(outputs_encoded_mu.cpu().detach().numpy())

train_labels = np.array([1])
scores = torch.norm(sample, dim=1) ** 2
scores = scores.detach().numpy()

predicted_labels = np.zeros_like(train_labels)
for i in range(scores.shape[0]):
    if(np.log(scores[i]) > np.log(scores).mean()):
        predicted_labels[i]=1
    else:
        predicted_labels[i]=0



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
np.save(_RESULTS_DIR+file_index+"_mat", mat)
print("max val:", mat.max())
print("min val:", mat.min())
print("mean val:", mat.mean())
mat = (255.0 *(mat - mat.min())/(mat.max() - mat.min())).astype('uint8')

# plt.imshow(mat, interpolation='nearest')

# imgplot = plt.imshow(mat, interpolation="nearest")
# plt.axis('off')
plt.imsave(_RESULTS_DIR+file_index+"_heatmap.png", mat)
# plt.colorbar()
# default colormap for plt is virdis
plt.close()

# sys.exit()
