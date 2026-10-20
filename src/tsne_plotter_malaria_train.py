import tkinter
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting
from matplotlib import cm
import numpy as np
import click
import torch
import logging
import random
import numpy as np
import PIL
from PIL import Image

from utils.config import Config
from utils.visualization.plot_images_grid import plot_images_grid, plot_matrix_grid
from DeepSAD import DeepSAD
from datasets.main import load_dataset
import torchvision.transforms as transforms


dataset_name = 'malaria_dataset'
data_path = "../../nanofibre/"
# net_name = "nanofibre_vae"
net_name = "cifar10_LeNet"
xp_path = _DATA_DIR
normal_class = 0
n_known_outlier_classes = 1
known_outlier_class = 1
ratio_known_normal = 0
ratio_known_outlier = 0.2
ratio_pollution = 0.1
seed = 0
load_model = os.path.join(_RESULTS_DIR, "model.tar")

# load_model = os.path.join(_RESULTS_DIR, "model.tar")
eta = 1.0
recon_param = 1.0
# device = "cuda:7"
device = "cpu"

deepSAD = DeepSAD(eta, recon_param, xp_path=xp_path)
deepSAD.set_network(net_name)

deepSAD.load_model(model_path=load_model, load_ae=True, map_location=device)

deepSAD.ae_net = deepSAD.ae_net.to(device)
deepSAD.ae_net.eval()

dataset = load_dataset(dataset_name, data_path, normal_class, known_outlier_class, n_known_outlier_classes,
                        ratio_known_normal, ratio_known_outlier, ratio_pollution,
                        random_state=np.random.RandomState(seed), length=5000)


train_labels = dataset.train_set.targets
print(type(train_labels))
#### Add the tranforms used for datasets here

# for i in range(len(dataset.test_set.tensor_batch)):
#     dataset.test_set.tensor_batch[i] = PIL.ImageOps.grayscale(dataset.test_set.tensor_batch[i])

print(dataset.train_set.tensor_batch[0])

# t1 = transforms.Compose([
# transforms.Resize((32, 32)),
# transforms.ToTensor()])

# final_test_data = torch.stack([I for I in dataset.test_set.tensor_batch])
final_train_data = torch.stack([I for I in dataset.train_set.tensor_batch])
final_train_data = final_train_data
train_labels = train_labels
print(final_train_data.shape)
final_train_data = final_train_data.to(device)

outputs_encoded_mu, outputs_encoded_beta, outputs_recons_mu, outputs_recons_beta, sample = deepSAD.ae_net(
    final_train_data)

print(outputs_encoded_mu.shape)

np.save("tsne_train_feats_E.npy", outputs_encoded_mu.cpu().detach().numpy())
np.save("train_labels_E.npy", np.array(train_labels))

outputs_encoded_mu = np.load("tsne_train_feats_E.npy")
train_labels = np.load("train_labels_E.npy")

train_labels = np.array(train_labels)

# tsne = TSNE(2, verbose=1)
tsne = TSNE(3, verbose=1)
tsne_proj = tsne.fit_transform(outputs_encoded_mu)
# tsne_proj = tsne.fit_transform(outputs_encoded_mu.cpu().detach().numpy())
# print(tsne_proj)
# Plot those points as a scatter plot and label them based on the pred labels
# cmap = cm.get_cmap('tab20')

######################
# fig, ax = plt.subplots(figsize=(8,8))
# # ax = fig.add_subplot(111, projection='3d')
# num_categories = 2
# for lab in range(num_categories):
#     indices = test_labels==lab
#     ax.scatter(tsne_proj[indices,0], tsne_proj[indices,1], label = lab ,alpha=0.5)
# ax.legend(fontsize='large', markerscale=2)
######################

########################
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax = fig.add_subplot(111, projection='3d')
num_categories = 2
for lab in range(num_categories):
    indices = train_labels==lab
    ax.scatter(tsne_proj[indices,0], tsne_proj[indices,1],tsne_proj[indices,2], label = lab ,alpha=0.5)
ax.legend(fontsize='large', markerscale=2)

# import pickle
# pickle.dump(fig, open('FigureObject.fig.pickle', 'wb'))

plt.show()
########################

plt.savefig(os.path.join(_DATA_DIR, "tsne_train_3d_1_train.png"))
# plt.savefig(os.path.join(_DATA_DIR, "tsne_test_3d_1.fig"))