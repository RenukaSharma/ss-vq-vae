import click
import torch
import logging
import random
import numpy as np

from utils.config import Config
from utils.visualization.plot_images_grid import plot_images_grid
from DeepSAD import DeepSAD
from datasets.main import load_dataset

import sys
import argparse
# sys.path.append('../')

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import time

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances as euc
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import scipy.io as sio

from PIL import Image

class CIFAR10_LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        # self.rep_dim = 128
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 32, 5, bias=False, padding=2)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(128 * 4 * 4, 128, bias=False)
        self.bn1d1 = torch.nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 16, bias=False)
        self.bn1d2 = torch.nn.BatchNorm1d(16)
        self.fc3 = nn.Linear(16, 1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.leaky_relu(self.bn1d1(x))
        x = self.fc2(x)
        x = F.leaky_relu(self.bn1d2(x))
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x


class MNIST_LeNet(nn.Module):
    def __init__(self, rep_dim=32):
        super().__init__()
        self.rep_dim = rep_dim
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(4 * 7 * 7, self.rep_dim, bias=False)
        self.bn1d1 = torch.nn.BatchNorm1d(self.rep_dim)
        self.fc2 = nn.Linear(self.rep_dim, 8, bias=False)
        self.bn1d2 = torch.nn.BatchNorm1d(8)
        self.fc3 = nn.Linear(8, 1, bias=False)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = x.view(int(x.size(0)), -1)
        x = self.fc1(x)
        x = F.leaky_relu(self.bn1d1(x))
        x = self.fc2(x)
        x = F.leaky_relu(self.bn1d2(x))
        x = self.fc3(x)
        x = torch.sigmnoid(x)
        return x


class FMNIST_LeNet(nn.Module):
    def __init__(self, rep_dim=64):
        super().__init__()
        self.rep_dim = rep_dim
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(1, 16, 5, bias=False, padding=2)
        self.bn2d1 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(16, 32, 5, bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(32 * 7 * 7, 128, bias=False)
        self.bn1d1 = nn.BatchNorm1d(128, eps=1e-04, affine=False)
        self.fc2 = nn.Linear(128, self.rep_dim, bias=False)
        self.bn1d2 = nn.BatchNorm1d(self.rep_dim, eps=1e-04, affine=False)
        self.fc3 = nn.Linear(self.rep_dim, 16, bias=False)
        self.bn1d3 = nn.BatchNorm1d(16, eps=1e-04, affine=False)
        self.fc4 = nn.Linear(16, 1, bias=False)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = x.view(int(x.size(0)), -1)
        x = F.leaky_relu(self.bn1d1(self.fc1(x)))
        x = self.fc2(x)
        x = F.leaky_relu(self.bn1d2(x))
        x = self.fc3(x)
        x = F.leaky_relu(self.bn1d3(x))
        x = self.fc4(x)
        x = torch.sigmnoid(x)
        return x


################################################################################
# Settings
################################################################################
# dataset_name, data_path, normal_class, known_outlier_class, n_known_outlier_classes,
						# ratio_known_normal, ratio_known_outlier, ratio_pollution,
						# random_state=np.random.RandomState(cfg.settings['seed'])

parser = argparse.ArgumentParser()
parser.add_argument('dataset_name', default='cifarr10')
parser.add_argument('net_name', default='cifar10_LeNet')
parser.add_argument('xp_path')
parser.add_argument('data_path')
parser.add_argument('normal_class', default=3)
parser.add_argument('known_outlier_class', default=1)
parser.add_argument('n_known_outlier_classes', default=1)
parser.add_argument('ratio_known_normal', default=0)
parser.add_argument('ratio_known_outlier', default=0)
parser.add_argument('ratio_pollution', default=0.0)
parser.add_argument('seed', default=0)
parser.add_argument('eta', default=1)

# @click.argument('dataset_name', type=click.Choice(['mnist', 'fmnist', 'cifar10', 'malaria_dataset', 'arrhythmia', 'cardio', 'satellite',
#                                                    'satimage-2', 'shuttle', 'thyroid']))
# @click.argument('net_name', type=click.Choice(['mnist_LeNet', 'fmnist_LeNet', 'cifar10_LeNet', 'malaria_net', 'arrhythmia_mlp',
#                                                'cardio_mlp', 'satellite_mlp', 'satimage-2_mlp', 'shuttle_mlp',
#                                                'thyroid_mlp']))
# @click.argument('xp_path', type=click.Path(exists=True))
# @click.argument('data_path', type=click.Path(exists=True))

# @click.option('--eta', type=float, default=1.0, help='Deep SAD hyperparameter eta (must be 0 < eta).')
# @click.option('--ratio_known_normal', type=float, default=0.0,
#               help='Ratio of known (labeled) normal training examples.')
# @click.option('--ratio_known_outlier', type=float, default=0.0,
#               help='Ratio of known (labeled) anomalous training examples.')
# @click.option('--ratio_pollution', type=float, default=0.0,
#               help='Pollution ratio of unlabeled training data with unknown (unlabeled) anomalies.')
# @click.option('--device', type=str, default='cuda', help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')
# @click.option('--seed', type=int, default=-1, help='Set seed. If -1, use randomization.')
# @click.option('--optimizer_name', type=click.Choice(['adam']), default='adam',
#               help='Name of the optimizer to use for Deep SAD network training.')
# @click.option('--lr', type=float, default=0.001,
#               help='Initial learning rate for Deep SAD network training. Default=0.001')
# @click.option('--n_epochs', type=int, default=50, help='Number of epochs to train.')
# @click.option('--lr_milestone', type=int, default=0, multiple=True,
#               help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
# @click.option('--batch_size', type=int, default=128, help='Batch size for mini-batch training.')
# @click.option('--weight_decay', type=float, default=1e-6,
#               help='Weight decay (L2 penalty) hyperparameter for Deep SAD objective.')

# @click.option('--num_threads', type=int, default=0,
#               help='Number of threads used for parallelizing CPU operations. 0 means that all resources are used.')
# @click.option('--n_jobs_dataloader', type=int, default=0,
#               help='Number of workers for data loading. 0 means that the data will be loaded in the main process.')
# @click.option('--normal_class', type=int, default=0,
#               help='Specify the normal class of the dataset (all other classes are considered anomalous).')
# @click.option('--known_outlier_class', type=int, default=1,
#               help='Specify the known outlier class of the dataset for semi-supervised anomaly detection.')
# @click.option('--n_known_outlier_classes', type=int, default=0,
#               help='Number of known outlier classes.'
#                    'If 0, no anomalies are known.'
#                    'If 1, outlier class as specified in --known_outlier_class option.'
#                    'If > 1, the specified number of outlier classes will be sampled at random.')


"""
Deep SAD, a method for deep semi-supervised anomaly detection.

:arg DATASET_NAME: Name of the dataset to load.
:arg NET_NAME: Name of the neural network to use.
:arg XP_PATH: Export path for logging the experiment.
:arg DATA_PATH: Root path of data.
"""
args = parser.parse_args()

log_file = args.xp_path + '/log.txt'

# Print paths
print('Log file is %s' % log_file)
print('Data path is %s' % args.data_path)
print('Export path is %s' % args.xp_path)

# Print experimental setup
print('Dataset: %s' % args.dataset_name)
print('Normal class: %d' % int(args.normal_class))
print('Ratio of labeled normal train samples: %.2f' %
			float(args.ratio_known_normal))
print('Ratio of labeled anomalous samples: %.2f' %
			float(args.ratio_known_outlier))
print('Pollution ratio of unlabeled train data: %.2f' %
			float(args.ratio_pollution))
if int(args.n_known_outlier_classes) == 1:
	print('Known anomaly class: %d' % int(args.known_outlier_class))
else:
	print('Number of known anomaly classes: %d' %
				int(args.n_known_outlier_classes))
print('Network: %s' % args.net_name)

# # If specified, load experiment config from JSON-file
# if load_config:
# 	cfg.load_config(import_json=load_config)
# 	print('Loaded configuration from %s.' % load_config)

# Print model configuration
print('Eta-parameter: %.2f' % float(args.eta))

# Set seed
if int(args.eta) != -1:
	random.seed(int(args.eta))
	np.random.seed(int(args.eta))
	torch.manual_seed(int(args.eta))
	torch.cuda.manual_seed(int(args.eta))
	torch.backends.cudnn.deterministic = True
	print('Set seed to %d.' % int(args.eta))

# Default device to 'cpu' if cuda is not available
# if not torch.cuda.is_available():
# 	device = 'cpu'
# 	print("CUDA not available")
# else:
# 	device='cuda'
device='cpu'
# Set the number of threads used for parallelizing CPU operations
num_threads=1
if num_threads > 0:
	torch.set_num_threads(num_threads)
# print('Computation device: %s' % args.device)
# print('Number of threads: %d' % args.num_threads)
# print('Number of dataloader workers: %d' % args.n_jobs_dataloader)

# Load data
dataset_name= args.dataset_name
data_path= args.data_path
normal_class= int(args.normal_class)
known_outlier_class=int(args.known_outlier_class)
n_known_outlier_classes=int(args.n_known_outlier_classes)
ratio_known_normal=float(args.ratio_known_normal)
ratio_known_outlier=float(args.ratio_known_outlier)
ratio_pollution=float(args.ratio_pollution)

dataset = load_dataset(dataset_name, data_path, normal_class, known_outlier_class,n_known_outlier_classes ,ratio_known_normal, ratio_known_outlier, ratio_pollution, random_state=np.random.RandomState(int(args.seed)))
# Log random sample of known anomaly classes if more than 1 class
if n_known_outlier_classes > 1:
	print('Known anomaly classes: %s' %(dataset.known_outlier_classes,))

	if(dataset_name == 'mnist'):
		net = MNIST_LeNet()
	if(dataset_name == 'fmnist'):
		net = FMNIST_LeNet()
	if(dataset_name == 'cifar10'):
		net = CIFAR10_LeNet()
	
	if device=='cuda':
		net= net.cuda()

	tr, ts = dataset.loaders(
		# batch_size=args.batch_size, num_workers=1, shuffle_train=True, shuffle_test=False)
		batch_size=128, num_workers=1, shuffle_train=True, shuffle_test=False)

	max_val = 0
	optimizer = optim.Adam(net.parameters(), lr=0.001,
							weight_decay=1e-6, amsgrad=False)
	print(len(tr))
	t = time.time()
	loss = nn.BCELoss(reduction='none')
	auc_list = []
	epochs=350
	for e in range(epochs):
		total = 0
		net.train()
		for data in tr:
			inputs, label, _, _ = data
			label = label.type(torch.FloatTensor)
			# print(ind.shape)
			# nump_dat = inputs.cpu().numpy()
			# input_label = label.cpu().numpy()
			# loss = nn.BCELoss()
			if device=='cuda':
				out = net(inputs.cuda())
			else:
				out = net(inputs)
			print("Output from the network is:",out)
			if device=='cuda':
				ll = loss(out, (label.cuda()).unsqueeze_(-1))
			else:
				ll = loss(out, (label).unsqueeze_(-1))
			print("The loss l1 is depicted as",ll)
			print("the label is", label)
			if device=='cuda':
				ll = ll/(label*(ratio_known_outlier-1)+1).cuda()
			else:
				ll = ll/(label*(ratio_known_outlier-1)+1)
			
			ll = torch.mean(ll)
			print("Mean loss is:", ll)
			ll.backward()
			optimizer.step()
			total += ll.item()
			# indx = ind.cpu().numpy()
			# print(ll)
			# print()
		show_loss=True
		if(show_loss):
			print("loss for epochs %d %f" % (e+1, total/len(tr)))
		# data_train = nump_dat.reshape(-1,28*28)
		# sio.savemat('mnist_mat/mnist_train_%d.mat'%(c),{"train_points": data, "train_labels": input_label})
		# continue
		net.eval()
		packet = []
		checkpoint=50
		if((e+1) % checkpoint != 0):
			continue
		with torch.no_grad():
			for data in ts:
				inputs, label, ind, num_labels = data
				label = label.type(torch.FloatTensor)
				if device=='cuda':
					score = net(inputs.cuda())
				else:
					score = net(inputs)
				packet += list(zip(label.cpu().numpy().tolist(),
									score.cpu().numpy().tolist()))
		label, score = zip(*packet)
		label = np.asarray(label)
		score = np.asarray(score)
		auc = roc_auc_score(label, score)
		if(auc > max_val):
			max_val = auc
		print("roc_score %f after %f epochs" % (auc, e+1))
	with torch.no_grad():
		for data in ts:
			inputs, label, _, _ = data
			label = label.type(torch.FloatTensor)
			if device=='cuda':
					score = net(inputs.cuda())
			else:
				score = net(inputs)
			packet += list(zip(label.cpu().numpy().tolist(),
								score.cpu().numpy().tolist()))
	label, score = zip(*packet)
	label = np.asarray(label)
	score = np.asarray(score)
	auc = roc_auc_score(label, score)
	if(auc > max_val):
		max_val = auc
	print("final auc %f max auc %f" % (auc, max_val))
	print(time.time()-t)

