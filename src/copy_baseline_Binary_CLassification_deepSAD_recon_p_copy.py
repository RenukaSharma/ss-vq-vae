import sys
import argparse
# sys.path.append('../')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import time
import random
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances as euc
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import scipy.io as sio
# from utils.config import Config
# from utils.visualization.plot_images_grid import plot_images_grid
# from deepSVDD import DeepSVDD
from datasets.main import load_dataset
from PIL import Image
# np.random.seed(0)
# for c in range(10):



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


class Malaria_net(nn.Module):

    def __init__(self):
        super().__init__()

        self.rep_dim = 128
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 32, 3, bias=False, padding=0,stride=2)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 64, 3, bias=False, padding=0)
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(64, 128, 3, bias=False, padding=0)
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.conv4= nn.Conv2d(128, self.rep_dim,1, bias=False)
        self.fc1 = nn.Linear(128 * 4 * 4, self.rep_dim, bias=False)
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
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.leaky_relu(self.bn1d1(x))
        x = self.fc2(x)
        x = F.leaky_relu(self.bn1d2(x))
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x


parser = argparse.ArgumentParser()
parser.add_argument("--dataset",default="cifar10",required=False,help="choose from malaria_dataset and cifar10")
parser.add_argument("--normal_class",default=3,type=int,required=False,help="Normal Class for the experiment")
parser.add_argument("--supervision_percentage",default=0.0,type=float,required=False,help="supervision percentage")
parser.add_argument("--epochs",default=100,type=int,required=False,help="number of epochs to run")
parser.add_argument("--batch_size",default=200,type=int,required=False,help="batch_size")
parser.add_argument("--checkpoint",default=5,type=int,required=False,help="number of epochs after which to check AUC score")
parser.add_argument("--supervision_class",default="all",required=False,help="supervision class")
parser.add_argument("--show_loss",default=1,type=int,required=False,help="Show loss after every epoch")
@click.option('--ratio_known_normal', type=float, default=0.0,
              help='Ratio of known (labeled) normal training examples.')
@click.option('--ratio_known_outlier', type=float, default=0.0,
              help='Ratio of known (labeled) anomalous training examples.')
@click.option('--ratio_pollution', type=float, default=0.0,
              help='Pollution ratio of unlabeled training data with unknown (unlabeled) anomalies.')
@click.option('--n_jobs_dataloader', type=int, default=0,
              help='Number of workers for data loading. 0 means that the data will be loaded in the main process.')
@click.option('--normal_class', type=int, default=0,
              help='Specify the normal class of the dataset (all other classes are considered anomalous).')
@click.option('--known_outlier_class', type=int, default=1,
              help='Specify the known outlier class of the dataset for semi-supervised anomaly detection.')
@click.option('--n_known_outlier_classes', type=int, default=0,
              help='Number of known outlier classes.'
                   'If 0, no anomalies are known.'
                   'If 1, outlier class as specified in --known_outlier_class option.'
                   'If > 1, the specified number of outlier classes will be sampled at random.')
@click.option('--seed', type=int, default=-1, help='Set seed. If -1, use randomization.')

args = parser.parse_args()

##############################
#### Preparing Parameters ####
##############################
epochs=args.epochs
checkpoint = args.checkpoint
if(args.dataset=="cifar10"):
	normal_class = args.normal_class

	# sp_perc = 0.05
	if(args.supervision_percentage>0.0):
		sp_perc = args.supervision_percentage
		if(args.supervision_class=="none"):
			sp_class=-1
			sp_div=1
		elif(args.supervision_class=="all"):
			sp_class=10
			sp_div = 9
		else:
			semi_sp_imp=[]
			for j in args.supervision_class:
				sp_class.append(int(j))
			sp_div = len(semi_sp_imp)
	dataset_name = 'cifar10'
	data_path = '../data'
	# dataset = load_dataset(dataset_name, data_path, normal_class,semi_sp_class=sp_class,semi_sp_impure_frac=sp_perc/sp_div)
	dataset = load_dataset(dataset_name, data_path, normal_class, known_outlier_class, n_known_outlier_classes,
                           ratio_known_normal, ratio_known_outlier, ratio_pollution,
                           random_state=np.random.RandomState(cfg.settings['seed']))
	net = CIFAR10_LeNet().cuda()
if(args.dataset=="mnist"):
	normal_class = args.normal_class

	# sp_perc = 0.05
	if(args.supervision_percentage>0.0):
		sp_perc = args.supervision_percentage
		if(args.supervision_class=="none"):
			sp_class=-1
			sp_div=1
		elif(args.supervision_class=="all"):
			sp_class=10
			sp_div = 9
		else:
			semi_sp_imp=[]
			for j in args.supervision_class:
				sp_class.append(int(j))
			sp_div = len(semi_sp_imp)
	dataset_name = 'mnist'
	data_path = '../data'
	dataset = load_dataset(dataset_name, data_path, normal_class,semi_sp_class=sp_class,semi_sp_impure_frac=sp_perc/sp_div)
	net = MNIST_LeNet().cuda()
if(args.dataset=="fmnist"):
	normal_class = args.normal_class

	# sp_perc = 0.05
	if(args.supervision_percentage>0.0):
		sp_perc = args.supervision_percentage
		if(args.supervision_class=="none"):
			sp_class=-1
			sp_div=1
		elif(args.supervision_class=="all"):
			sp_class=10
			sp_div = 9
		else:
			semi_sp_imp=[]
			for j in args.supervision_class:
				sp_class.append(int(j))
			sp_div = len(semi_sp_imp)
	dataset_name = 'fmnist'
	data_path = '../data'
	dataset = load_dataset(dataset_name, data_path, normal_class,semi_sp_class=sp_class,semi_sp_impure_frac=sp_perc/sp_div)
	net = FMNIST_LeNet().cuda()
elif(args.dataset=="malaria_dataset"):
	sp_perc = args.supervision_percentage
	dataset_name = 'malaria_dataset'
	data_path = '../data/malaria_AD_dataset_500'
	dataset = load_dataset(dataset_name, data_path,args.normal_class,semi_sp_class=10,semi_sp_impure_frac=sp_perc)
	net = Malaria_net().cuda()
else:
	print("Unknown Dataset")
	exit()
tr,ts = dataset.loaders(batch_size=args.batch_size, num_workers=1,shuffle_train=True, shuffle_test=False)
max_val = 0
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-6,amsgrad=False)
print(len(tr))
t=time.time()
loss = nn.BCELoss(reduction='none')
auc_list=[]
for e in range(epochs):
	total=0
	net.train()
	for data in tr:
		inputs, label, ind,num_labels = data
		label = label.type(torch.FloatTensor)
		# print(ind.shape)
		# nump_dat = inputs.cpu().numpy()
		# input_label = label.cpu().numpy()
		# loss = nn.BCELoss()
		out = net(inputs.cuda())
		ll = loss(out,(label.cuda()).unsqueeze_(-1))
		ll = ll/(label*(sp_perc-1)+1).cuda()
		ll = torch.mean(ll)
		ll.backward()
		optimizer.step()
		total+=ll.item()
		# indx = ind.cpu().numpy()
		# print(ll)
		# print()
	if(args.show_loss):
		print("loss for epochs %d %f"%(e+1,total/len(tr)))
	# data_train = nump_dat.reshape(-1,28*28)
	# sio.savemat('mnist_mat/mnist_train_%d.mat'%(c),{"train_points": data, "train_labels": input_label})
	# continue
	net.eval()
	packet=[]
	if((e+1)%checkpoint!=0):
		continue
	with torch.no_grad():
		for data in ts:
			inputs, label, ind,num_labels = data
			label = label.type(torch.FloatTensor)
			score = net(inputs.cuda())
			packet += list(zip(label.cpu().numpy().tolist(),score.cpu().numpy().tolist()))
	label,score = zip(*packet)
	label = np.asarray(label)
	score = np.asarray(score)
	auc = roc_auc_score(label,score)
	if(auc>max_val):
		max_val=auc
	print("roc_score %f after %f epochs"%(auc,e+1))
with torch.no_grad():
	for data in ts:
		inputs, label, ind,num_labels = data
		label = label.type(torch.FloatTensor)
		score = net(inputs.cuda())
		packet += list(zip(label.cpu().numpy().tolist(),score.cpu().numpy().tolist()))
label,score = zip(*packet)
label = np.asarray(label)
score = np.asarray(score)
auc = roc_auc_score(label,score)
if(auc>max_val):
	max_val=auc
print("final auc %f max auc %f"%(auc,max_val))
print(time.time()-t)