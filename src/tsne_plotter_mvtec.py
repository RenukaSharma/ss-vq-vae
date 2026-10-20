from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
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


dataset_name = 'mvtec'
# data_path = _DATA_DIR
data_path = _DATA_DIR
net_name = "cifar10_LeNet"
xp_path = _DATA_DIR

normal_class = 0
n_known_outlier_classes = 1
known_outlier_class = 1
ratio_known_normal = 0
ratio_known_outlier = 0.2
ratio_pollution = 0.2
seed = 0
# load_model = os.path.join(_RESULTS_DIR, "model.tar")
# load_model = os.path.join(_RESULTS_DIR, "model.tar")
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

test_labels = dataset.test_set.targets
print(type(test_labels))
#### Add the tranforms used for datasets here

# for i in range(len(dataset.test_set.data)):
#     dataset.test_set.data[i] = PIL.ImageOps.grayscale(dataset.test_set.data[i])

print(dataset.test_set.data[0])

t1 = transforms.Compose([
transforms.Resize((32, 32)),
transforms.ToTensor()])

final_test_data = torch.stack([t1(I) for I in dataset.test_set.data])
final_test_data = final_test_data
test_labels = test_labels
print(final_test_data.shape) # printing the shape
final_test_data = final_test_data.to(device)

outputs_encoded_mu, outputs_encoded_beta, outputs_recons_mu, outputs_recons_beta, sample = deepSAD.ae_net(final_test_data)

print(outputs_encoded_mu.shape)

np.save("tsne_feats.npy", outputs_encoded_mu.cpu().detach().numpy())
np.save("test_labels.npy", np.array(test_labels))

outputs_encoded_mu = np.load("tsne_feats.npy")
test_labels = np.load("test_labels.npy")

test_labels = np.array(test_labels)

tsne = TSNE(2, verbose=1)
tsne_proj = tsne.fit_transform(outputs_encoded_mu)
# tsne_proj = tsne.fit_transform(outputs_encoded_mu.cpu().detach().numpy())
# print(tsne_proj)
# Plot those points as a scatter plot and label them based on the pred labels
# cmap = cm.get_cmap('tab20')
fig, ax = plt.subplots(figsize=(8,8))

num_categories = 2
for lab in range(num_categories):
    indices = test_labels==lab
    ax.scatter(tsne_proj[indices,0], tsne_proj[indices,1], label = lab ,alpha=0.5)
ax.legend(fontsize='large', markerscale=2)
# plt.savefig(os.path.join(_DATA_DIR, "mvtec_carpet_tsne_test.png"))
# plt.savefig(os.path.join(_DATA_DIR, "mvtec_bottle_tsne_test.png"))
# plt.savefig(os.path.join(_DATA_DIR, "mvtec_leather_tsne_test.png"))
plt.savefig(os.path.join(_DATA_DIR, "mvtec_tile_tsne_test.png"))