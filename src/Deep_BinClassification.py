import json
import torch
import torch.nn as nn
import torch.optim as optim
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score


import numpy as np

from base.base_dataset import BaseADDataset
# from networks.main_non-DeepSAD import build_network, build_autoencoder
from optim.DeepSAD_trainer import DeepSADTrainer
from optim.ae_trainer import AETrainer

import logging
from base.base_net import BaseNet

class MNIST_LeNet(BaseNet):

    def __init__(self, rep_dim=32):
        super().__init__()

        self.rep_dim = rep_dim           
        
        self.conv1 = nn.Conv2d(1, 8, 4, bias=True, stride=2, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d1 = nn.BatchNorm2d(8, eps=1e-04, affine=True)

        self.conv2 = nn.Conv2d(8, 16, 4, bias=True, stride=2, padding=1)
        nn.init.xavier_uniform_(self.conv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d2 = nn.BatchNorm2d(16, eps=1e-04, affine=True)

        self.conv3 = nn.Conv2d(16, 32, 4, bias=True, stride=2, padding=1)
        nn.init.xavier_uniform_(self.conv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d3 = nn.BatchNorm2d(32, eps=1e-04, affine=True)

        self.fc1 = nn.Linear(32 * 3 * 3, self.rep_dim, bias=True)
        nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        
        self.fc2 = nn.Linear(self.rep_dim, 2, bias=True)
        nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.conv1(x)
        x = F.leaky_relu(self.bn2d1(x))
        x = self.conv2(x)
        x = F.leaky_relu(self.bn2d2(x))
        x = self.conv3(x)
        x = F.leaky_relu(self.bn2d3(x))
        x = x.view(int(x.size(0)), -1)        
        x = F.leaky_relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        
        return x

class FMNIST_LeNet(BaseNet):

    def __init__(self, rep_dim=128):
        super().__init__()

        self.rep_dim = rep_dim

        # for mu
        self.conv1 = nn.Conv2d(1, 16, 4, bias=True, stride=2, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d1 = nn.BatchNorm2d(16, eps=1e-04, affine=True)

        self.conv2 = nn.Conv2d(16, 32, 4, bias=True, stride=2, padding=1)
        nn.init.xavier_uniform_(self.conv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d2 = nn.BatchNorm2d(32, eps=1e-04, affine=True)

        self.conv3 = nn.Conv2d(32, 64, 4, bias=True, stride=2, padding=1)
        nn.init.xavier_uniform_(self.conv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d3 = nn.BatchNorm2d(64, eps=1e-04, affine=True)

        self.fc1 = nn.Linear(64 * 3 * 3, self.rep_dim, bias=True)
        nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('leaky_relu'))

        self.fc2 = nn.Linear(self.rep_dim, 2, bias=True)
        nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, x):
        logger = logging.getLogger()
        x = x.view(-1, 1, 28, 28)
        x = self.conv1(x)
        x = F.leaky_relu(self.bn2d1(x))
        x = self.conv2(x)
        x = F.leaky_relu(self.bn2d2(x))
        x = self.conv3(x)
        x = F.leaky_relu(self.bn2d3(x))
        x = x.view(int(x.size(0)), -1)
        
        x = (self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        
        return x #x_encoded

class CIFAR10_LeNet(BaseNet):

    def __init__(self, rep_dim=128):
        super().__init__()

        self.rep_dim = rep_dim
        
        self.pool = nn.MaxPool2d(2, 2)
        # for mu
        self.conv1 = nn.Conv2d(3, 32, 4, bias=False, stride=2, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)

        self.conv2 = nn.Conv2d(32, 64, 4, bias=False, stride=2, padding=1)
        nn.init.xavier_uniform_(self.conv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)

        self.conv3 = nn.Conv2d(64, 128, 4, bias=False, stride=2, padding=1)
        nn.init.xavier_uniform_(self.conv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)

        self.fc1 = nn.Linear(128 * 4 * 4, self.rep_dim, bias=False)
        nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('leaky_relu'))

        self.fc2 = nn.Linear(self.rep_dim, 2, bias=True)
        nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, x):
        
        x = x.view(-1, 3, 32, 32)
        x = self.conv1(x)
        x = F.leaky_relu(self.bn2d1(x))
        x = self.conv2(x)
        x = F.leaky_relu(self.bn2d2(x))
        x = self.conv3(x)
        x = F.leaky_relu(self.bn2d3(x))
        x = x.view(int(x.size(0)), -1)

        x = self.fc1(x)
        x = F.log_softmax(self.fc2(x), dim=1)

        return x


def build_autoencoder(net_name):
    ae_net = None
    if(net_name=='cifar10'):
        ae_net = CIFAR10_LeNet()

    if(net_name=='fmnist'):
        ae_net = FMNIST_LeNet()

    if(net_name=='mnist'):
        ae_net = MNIST_LeNet()
    
    return ae_net

class DeepSAD(object):
    """A class for the Deep SAD method.

    Attributes:
        eta: Deep SAD hyperparameter eta (must be 0 < eta).
        c: Hypersphere center c.
        net_name: A string indicating the name of the neural network to use.
        net: The neural network phi.
        trainer: DeepSADTrainer to train a Deep SAD model.
        optimizer_name: A string indicating the optimizer to use for training the Deep SAD network.
        ae_net: The autoencoder network corresponding to phi for network weights pretraining.
        ae_trainer: AETrainer to train an autoencoder in pretraining.
        ae_optimizer_name: A string indicating the optimizer to use for pretraining the autoencoder.
        results: A dictionary to save the results.
        ae_results: A dictionary to save the autoencoder results.
    """

    def __init__(self, eta: float = 1.0, recon_param: float = 0.5, xp_path: str = "abc"):
        """Inits DeepSAD with hyperparameter eta."""

        self.eta = eta
        self.recon_param = recon_param
        
        self.c = None  # hypersphere center c

        self.net_name = None
        self.net = None  # neural network phi

        self.trainer = None
        self.optimizer_name = None

        self.ae_net = None  # autoencoder network for pretraining
        self.dec_net = None
        self.ae_trainer = None
        self.ae_optimizer_name = None

        self.batch_size = 128
        self.n_jobs_dataloader =0

        self.xp_path = xp_path

        # self.device = 'cpu'

        self.results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None,
            'psnr' : None,            
            'test_scores': None
        }

        self.ae_results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None
        }

    

    def train(self, dataset: BaseADDataset, dataset_name: str = 'cifar10',optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 50,
              lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cpu',
              n_jobs_dataloader: int = 0 ):
        """Trains the Deep SAD model on the training data."""
        
        self.optimizer_name = optimizer_name
        logger = logging.getLogger()
        self.device=device
        # Get train data loader
        train_loader, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)
        # copy_dataset = copy.deepcopy(dataset)
        # Set device for network

        ae_net = build_autoencoder(dataset_name)
        self.ae_net = ae_net
        ae_net = ae_net.to(self.device)

        # Set optimizer (Adam optimizer for now)    

        optimizer = optim.Adam(ae_net.parameters(), lr=lr, weight_decay=weight_decay)

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=0.1)
        self.criterion = nn.CrossEntropyLoss()
        # self.ablation_type = ablation_type
        # Training
        logger.info('Starting training...')
        print("Starting training.....")
        start_time = time.time()
        # net.train()
        # dec_net.train()

        self.test_aucs = []
        self.latent_losses = []
        self.image_losses = []
        self.recon_losses = []
        self.losses = []

        for epoch in range(n_epochs):
            ae_net.train()
            # scheduler.step()
            # # scheduler_joint.step()
            # if epoch in self.lr_milestones:
            #     logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            epoch_loss = 0.0
            n_batches = 0
            recon_loss = 0.0
            kl_loss = 0.0
            image_loss = 0.0
            epoch_start_time = time.time()
            # sum_labels = torch.tensor(0.0)
            for data in train_loader:
                optimizer.zero_grad()

                inputs, labels, semi_targets, _ = data
                inputs, semi_targets = inputs.to(self.device), semi_targets.to(self.device)

                outputs = ae_net(inputs)
                # print("Unique sem-targets before tweaking:", torch.unique(semi_targets))
                semi_targets[semi_targets==1]=0
                semi_targets[semi_targets==-1]=1
                # print("Unique sem-targets before tweaking:", torch.unique(semi_targets))
                # print(semi_targets)
                # print(outputs)
                loss = self.criterion(outputs, semi_targets)
                loss.backward()
                # optimizer_joint.step()
                optimizer.step()
                epoch_loss += loss.item()
                
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            test_auc = self.test(dataset,  ae_net, xp_path=self.xp_path)
            # test_auc = 0
            logger.info(f'| Epoch: {epoch + 1:03}/{n_epochs:03} | Test AUC: {test_auc:.3f}s '
                        f'| Train Loss: {epoch_loss / n_batches:.6f} | Recon loss: {recon_loss / n_batches} | Latent loss: {kl_loss / n_batches} | Image loss: {image_loss / n_batches}')
            
            self.test_aucs.append(test_auc)
            self.losses.append(epoch_loss / n_batches)
            self.latent_losses.append(kl_loss / n_batches)
            self.image_losses.append(image_loss / n_batches)
            self.recon_losses.append(recon_loss / n_batches)

        self.train_time = time.time() - start_time
        logger.info('Training Time: {:.3f}s'.format(self.train_time))
        logger.info('Finished training.')

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot(self.test_aucs, label="test_auc")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(self.losses, label="loss")
        plt.plot(self.image_losses, label="image loss")
        plt.plot(self.latent_losses, label="latent loss")
        plt.legend()
        plt.savefig(self.xp_path + "/plots.png")

        # Get the model
        
        self.results['train_time'] = self.train_time
        # self.c = self.trainer.c.cpu().data.numpy().tolist()  # get as list

    def test(self, dataset: BaseADDataset, device: str ='cpu' ,n_jobs_dataloader: int = 0, xp_path: str ='../'):
        """Tests the Deep SAD model on the test data."""

        # if self.trainer is None:
        #     self.trainer = DeepSADTrainer(self.c, self.eta, device=device, n_jobs_dataloader=n_jobs_dataloader)

        logger = logging.getLogger()

        # Get test data loader
        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set device for network
        # self.device=device

        ae_net = self.ae_net.to(self.device)

        # Testing
        # logger.info('Starting testing...')
        epoch_loss = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []

        ae_net.eval()
        sum_labels = torch.tensor(0.0)
        totals = torch.tensor(0.0)
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, semi_targets, idx = data
                semi_targets[semi_targets==1]=0
                semi_targets[semi_targets==-1]=1
                
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                idx = idx.to(self.device)

                outputs = ae_net(inputs)
                # print("outputs size:", outputs.shape)
                _, pred = torch.max(outputs, 1)
                # print("Actual labels:", labels) 
                # print("Predicted output:", pred)

                # print("Labels size:", labels.shape)
                # print("Predicted label:", pred.shape)

                # print("Labels type:", type(labels))
                # print("Predicted type:", type(pred))

                # print("pred.unsqueeze(dim=0).shape=", pred.unsqueeze(dim=0).shape)
                # print("labels.shape=", labels.shape)

                # loss = self.criterion(pred.unsqueeze(dim=0).float(), labels)
                loss = self.criterion(outputs, labels)
                
                
                scores = pred
                # scores = outputs
                
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

                epoch_loss += loss.item()
                n_batches += 1

        # self.psnr = psnr
        
        self.test_time = time.time() - start_time
        self.test_scores = idx_label_score

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        # print(scores)
        self.test_auc = roc_auc_score(labels, scores)

        

        # Get results
        self.results['test_auc'] = self.test_auc
        self.results['test_time'] = self.test_time
                
        self.results['test_scores'] = self.test_scores
        return self.test_auc

    def save_model(self, export_model, save_ae=True):
        """Save Deep SAD model to export_model."""

        net_dict = self.net.state_dict()
        ae_net_dict = self.ae_net.state_dict() if save_ae else None

        torch.save({ 'c': self.c,
                    'net_dict': net_dict,
                    'ae_net_dict': ae_net_dict}, export_model)

    def load_model(self, model_path, load_ae=False, map_location='cpu'):
        """Load Deep SAD model from model_path."""

        model_dict = torch.load(model_path, map_location=map_location)

        self.c = model_dict['c']
        self.net.load_state_dict(model_dict['net_dict'])

        # load autoencoder parameters if specified
        if load_ae:
            if self.ae_net is None:
                self.ae_net = build_autoencoder(self.net_name)
            self.ae_net.load_state_dict(model_dict['ae_net_dict'])

    def save_results(self, export_json):
        """Save results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.results, fp)

    def save_ae_results(self, export_json):
        """Save autoencoder results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.ae_results, fp)
