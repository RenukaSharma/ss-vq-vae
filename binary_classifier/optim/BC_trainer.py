from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score

import logging
import time
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import copy
import sys
from PIL import Image
import matplotlib.pyplot as plt


# import cv2


def log_gg(x, mu, alpha, beta, delta, tau, eps):
    temp = (
            - (((torch.abs(mu - x) / (delta + alpha)) ** 2 + eps) ** (0.5 * (tau + beta)))
            + (torch.log(tau + beta))
            - (torch.lgamma(1 / (tau + beta)))
            - (torch.log(delta + alpha))
            - (torch.log(torch.tensor(2.0)))
    )
    return torch.mean(temp, dim=tuple(range(1, mu.dim())))


class BCTrainer(BaseTrainer):

    def __init__(self, c, eta: float, dataset_name: str = 'cifar10', optimizer_name: str = 'adam', lr: float = 0.001,
                 n_epochs: int = 150,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0, recon_param: float = 0.5, eps: float = 0.1, tau: float = 0.1,
                 delta: float = 0.1, const: float = 0.1, xp_path: str = "abc"):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        # Deep SAD parameters
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.eta = eta
        self.dataset_name = dataset_name
        self.xp_path = xp_path

        # Optimization parameters
        self.eps = eps

        if (dataset_name == 'mnist'):
            self.rep_dim = 32
        if (dataset_name == 'fmnist'):
            self.rep_dim = 64
        if (dataset_name == 'cifar10'):
            self.rep_dim = 128

        self.batch_size = 128

        self.net_p_value = None
        self.net_c_value = None

        self.ae_net_p_value = None
        self.ae_net_c_value = None
        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.psnr = None
        self.test_scores = None
        self.recon_param = recon_param

        # small real-values
        self.tau = tau
        self.delta = delta
        self.const = const

    def train(self, dataset: BaseADDataset = 'cifar10', net: BaseNet = None, dec_net: BaseNet = None,
              ae_net: BaseNet = None, ablation_type='A'):

        logger = logging.getLogger()

        # Get train data loader
        train_loader, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)
        # copy_dataset = copy.deepcopy(dataset)
        # Set device for network

        ae_net = ae_net.to(self.device)

        # Set optimizer (Adam optimizer for now)    

        optimizer = optim.Adam(ae_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)
        self.criterion = nn.BCELoss()
        self.ablation_type = ablation_type
        # Training
        logger.info('Starting training...')
        start_time = time.time()
        # net.train()
        # dec_net.train()

        self.test_aucs = []
        self.latent_losses = []
        self.image_losses = []
        self.recon_losses = []
        self.losses = []

        for epoch in range(self.n_epochs):
            ae_net.train()
            scheduler.step()
            # scheduler_joint.step()
            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            epoch_loss = 0.0
            n_batches = 0
            recon_loss = 0.0
            kl_loss = 0.0
            image_loss = 0.0
            epoch_start_time = time.time()
            # sum_labels = torch.tensor(0.0)
            for data in train_loader:

                
                inputs, labels, semi_targets, _ = data  # labels was _ previously
                inputs, semi_targets = inputs.to(self.device), semi_targets.to(self.device)
                targets = (semi_targets < 0)
                targets = targets.type(torch.FloatTensor).to(self.device).reshape(-1, 1)
                # print(targets.shape, targets)

                outputs = ae_net(inputs)
                
                # print("out type:", type(outputs))
                # print("out size:", outputs.size())
                # print("target type:", type(targets))
                # print("target size:", targets.size())
                loss = self.criterion(outputs, targets)

                optimizer.zero_grad()
                # loss = torch.mean(losses)
                # loss = losses
                loss.backward()
                # optimizer_joint.step()
                optimizer.step()
                epoch_loss += loss.item()
                
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            test_auc = self.test(dataset, net, dec_net, ae_net, '../', verbose=False)
            # test_auc = 0
            logger.info(f'| Epoch: {epoch + 1:03}/{self.n_epochs:03} | Test AUC: {test_auc:.3f}s '
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
        plt.clf()

        return net, dec_net, ae_net

    def loss_A(self, data, ae_net):
        inputs, labels, semi_targets, _ = data  # labels was _ previously
        inputs, semi_targets = inputs.to(self.device), semi_targets.to(self.device)

        outputs_encoded_mu, outputs_encoded_alpha, outputs_encoded_beta, outputs_recons_mu, outputs_recons_alpha, outputs_recons_beta, sample = ae_net(inputs, ablation_type='A')

        standard_alpha = torch.ones_like(outputs_encoded_mu) * (2 ** 0.5)
        standard_beta = torch.ones_like(outputs_encoded_mu) * 2.0
        standard_recons_alpha = torch.ones_like(outputs_recons_mu) * (2 ** 0.5)
        standard_recons_beta = torch.ones_like(outputs_recons_mu) * 2.0
        dist_first = log_gg(sample, outputs_encoded_mu, outputs_encoded_alpha,
                            outputs_encoded_beta, self.delta, self.tau, self.eps)
        dist_second = 0.5 * (torch.norm(sample, dim=1) ** 2)
        dist_recon = (-1) * log_gg(inputs, outputs_recons_mu, outputs_recons_alpha,
                                   outputs_recons_beta,
                                   self.delta, self.tau, self.eps)

        dist = dist_first + dist_second + dist_recon
        latent = dist_first + dist_second
        latent_semisup = torch.where(semi_targets == 0, latent, self.eta * ((latent + self.eps) ** semi_targets.float()))
        recon_semisup = torch.where(semi_targets == 0, dist_recon, self.eta * ((dist_recon + self.eps) ** semi_targets.float()))
        # print("Beta: ", torch.mean(torch.exp(outputs_encoded_beta)))
        # print(torch.mean(dist_first), torch.mean(dist_second), torch.mean(dist_recon))
        losses = latent_semisup + recon_semisup
        # losses = torch.where(semi_targets == 0, dist, self.eta * ((dist + self.eps) ** semi_targets.float()))
        recon_loss = self.criterion(inputs, outputs_recons_mu)
        losses = torch.mean(losses)
        return losses, latent, dist_recon, recon_loss

    def loss_B(self, data, ae_net):
        inputs, labels, semi_targets, _ = data  # labels was _ previously
        inputs, semi_targets = inputs.to(self.device), semi_targets.to(self.device)

        outputs_encoded_mu, outputs_encoded_alpha, outputs_encoded_beta, outputs_recons_mu, outputs_recons_alpha, outputs_recons_beta, sample = ae_net(
            inputs, ablation_type='B')

        # latent = torch.norm(outputs_encoded_mu, dim=1)**2
        latent = torch.sum(sample**2, dim=1)
        standard_recons_alpha = torch.ones_like(outputs_recons_mu) * (2 ** 0.5)
        standard_recons_beta = torch.ones_like(outputs_recons_mu) * 2.0
        # image_space = torch.mean((inputs - outputs_recons_mu)**2, dim=(1, 2, 3))
        image_space = (-1) * log_gg(inputs, outputs_recons_mu, standard_recons_alpha,
                                   standard_recons_beta, 0, 0, 0)
        latent_semisup = torch.where(semi_targets == 0, latent, self.eta * ((latent + self.eps) ** semi_targets.float()))
        recon_semisup = torch.where(semi_targets == 0, image_space, self.eta * ((image_space + self.eps) ** semi_targets.float()))

        losses = recon_semisup + latent_semisup

        recon_loss = self.criterion(inputs, outputs_recons_mu)
        # losses = torch.where(semi_targets == 0, losses, self.eta * ((losses + self.eps) ** semi_targets.float()))
        losses = torch.mean(losses)
        return losses, latent, image_space, recon_loss

    def loss_C(self, data, ae_net):
        inputs, labels, semi_targets, _ = data  # labels was _ previously
        inputs, semi_targets = inputs.to(self.device), semi_targets.to(self.device)

        outputs_encoded_mu, outputs_encoded_alpha, outputs_encoded_beta, outputs_recons_mu, outputs_recons_alpha, outputs_recons_beta, sample = ae_net(inputs, ablation_type='A')

        # dist_first = log_gg(sample, outputs_encoded_mu, torch.exp(outputs_encoded_alpha),
        #                     torch.exp(outputs_encoded_beta), self.delta, self.tau, self.eps)
        # dist_second = 0.5 * (torch.norm(sample, dim=1) ** 2)
        latent = torch.sum(sample**2, dim=1)
        latent_semisup = torch.where(semi_targets == 0, latent, self.eta * ((latent + self.eps) ** semi_targets.float()))
        dist_recon = (-1) * log_gg(inputs, outputs_recons_mu, outputs_recons_alpha,
                                   outputs_recons_beta,
                                   self.delta, self.tau, self.eps)

        # dist = dist_first + dist_second + dist_recon
        # latent = self.recon_param*(dist_first + dist_second)
        # latent_semisup = torch.where(semi_targets == 0, latent, self.eta * ((latent + self.eps) ** semi_targets.float()))
        recon_semisup = torch.where(semi_targets == 0, dist_recon, self.eta * ((dist_recon + self.eps) ** semi_targets.float()))
        # print("Beta: ", torch.mean(torch.exp(outputs_encoded_beta)))
        # print(torch.mean(dist_first), torch.mean(dist_second), torch.mean(dist_recon))
        losses = latent_semisup + recon_semisup
        # losses = torch.where(semi_targets == 0, dist, self.eta * ((dist + self.eps) ** semi_targets.float()))
        recon_loss = self.criterion(inputs, outputs_recons_mu)
        losses = torch.mean(losses)
        return losses, latent, dist_recon, recon_loss
    
    def loss_D(self, data, ae_net):
        inputs, labels, semi_targets, _ = data  # labels was _ previously
        inputs, semi_targets = inputs.to(self.device), semi_targets.to(self.device)

        outputs_encoded_mu, outputs_encoded_alpha, outputs_encoded_beta, outputs_recons_mu, outputs_recons_alpha, outputs_recons_beta, sample = ae_net(
            inputs, ablation_type='D')

        image_space = torch.mean((inputs - outputs_recons_mu) ** 2, dim=(1, 2, 3))
        
        recon_loss = torch.mean(self.criterion(inputs, outputs_recons_mu),  dim=(1, 2, 3))
        # losses = recon_loss.copy()
        losses = torch.where(semi_targets == 0, recon_loss, self.eta * ((recon_loss + self.eps) ** semi_targets.float()))
        losses = torch.mean(losses)

        return losses, torch.zeros_like(image_space), recon_loss, image_space

    def loss_E(self, data, ae_net):
        inputs, labels, semi_targets, _ = data  # labels was _ previously
        inputs, semi_targets = inputs.to(self.device), semi_targets.to(self.device)

        outputs_encoded_mu, outputs_encoded_alpha, outputs_encoded_beta, outputs_recons_mu, outputs_recons_alpha, outputs_recons_beta, sample = ae_net(
            inputs, ablation_type='E')

        # image_space = torch.mean((inputs - outputs_recons_mu) ** 2, dim=(1, 2, 3))
        latent = torch.sum(sample**2, dim=1)

        losses = latent

        recon_loss = self.criterion(inputs, outputs_recons_mu)
        losses = torch.where(semi_targets == 0, losses, self.eta * ((losses + self.eps) ** semi_targets.float()))
        losses = torch.mean(losses)
        return losses, latent, torch.zeros_like(latent), recon_loss

    def loss_VAE(self, data, ae_net):
        """
        Gaussian Variational Autoencoder loss
        """
        def kl_g(mu, log_sigma):
            return 0.5 * (mu ** 2 + torch.exp(log_sigma) ** 2 - 1) - log_sigma

        inputs, labels, semi_targets, _ = data  # labels was _ previously
        inputs, semi_targets = inputs.to(self.device), semi_targets.to(self.device)

        outputs_encoded_mu, outputs_encoded_alpha, outputs_encoded_beta, outputs_recons_mu, outputs_recons_alpha, outputs_recons_beta, sample = ae_net(
            inputs, ablation_type='VAE')

        kl_loss = kl_g(outputs_encoded_mu, outputs_encoded_alpha)
        kl_loss = torch.sum(kl_loss, dim=1)
        standard_recons_beta = torch.ones_like(outputs_recons_mu) * 2.0
        # dist_recon = (-1) * log_gg(inputs, outputs_recons_mu, torch.exp(outputs_recons_alpha),
                                #    standard_recons_beta,
                                #    self.delta, 0.0, self.eps)

        kl_semisup = torch.where(semi_targets == 0, kl_loss,
                                     self.eta * ((kl_loss + self.eps) ** semi_targets.float()))
        

        # losses = kl_semisup + recon_semisup
        recon_loss = self.criterion(inputs, outputs_recons_mu)
        recon_loss = torch.mean(recon_loss, dim=(1, 2, 3))

        recon_semisup = torch.where(semi_targets == 0, recon_loss,
                                    self.eta * ((recon_loss + self.eps) ** semi_targets.float()))
        
        losses = kl_semisup + recon_semisup
        losses = torch.mean(losses)
        # losses = recon_loss
        # losses = torch.where(semi_targets == 0, losses, self.eta * ((losses + self.eps) ** semi_targets.float()))
        return losses, kl_loss, recon_loss, recon_loss

    def loss_test(self, data, ae_net):
        inputs, labels, semi_targets, _ = data  # labels was _ previously
        inputs, semi_targets = inputs.to(self.device), semi_targets.to(self.device)

        outputs_encoded_mu, outputs_encoded_alpha, outputs_encoded_beta, outputs_recons_mu, outputs_recons_alpha, outputs_recons_beta, sample = ae_net(
            inputs, ablation_type='test')

        # dist_first = (-1)*log_gg(sample, outputs_encoded_mu, outputs_encoded_alpha,
                            # outputs_encoded_beta, self.delta, self.tau, self.eps)
        dist_second = 0.5 * (torch.norm(sample, dim=1) ** 2)
        dist_recon = (-1) * log_gg(inputs, outputs_recons_mu, outputs_recons_alpha,
                                   outputs_recons_beta,
                                   self.delta, self.tau, self.eps)

        dist = dist_second + dist_recon
        latent = dist_second
        latent_semisup = torch.where(semi_targets == 0, latent,
                                     self.eta * ((latent + self.eps) ** semi_targets.float()))
        recon_semisup = torch.where(semi_targets == 0, dist_recon,
                                    self.eta * ((dist_recon + self.eps) ** semi_targets.float()))
        # print("Beta: ", torch.mean(torch.exp(outputs_encoded_beta)))
        # print(torch.mean(dist_first), torch.mean(dist_second), torch.mean(dist_recon))
        losses = latent_semisup + recon_semisup
        # losses = latent_semisup
        # losses = torch.where(semi_targets == 0, dist, self.eta * ((dist + self.eps) ** semi_targets.float()))
        recon_loss = self.criterion(inputs, outputs_recons_mu)
        losses = torch.mean(losses)
        return losses, latent, torch.zeros_like(latent), recon_loss

    def test(self, dataset: BaseADDataset = 'cifar10', net: BaseNet = None, dec_net: BaseNet = None,
             ae_net: BaseNet = None, xp_path: str = '../', verbose: bool = True):

        def psnr_fun(orig, recon):
            mse = np.mean((orig - recon) ** 2)
            if (mse == 0):
                return 100
            max_pixel = np.max(orig)  # 255.0
            psnr = 20 * np.log(max_pixel / np.sqrt(mse))
            return psnr

        logger = logging.getLogger()

        # Get test data loader
        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set device for network

        ae_net = ae_net.to(self.device)

        # Testing
        # logger.info('Starting testing...')
        epoch_loss = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []

        ae_net.eval()
        sum_labels = torch.tensor(0.0)
        totals = torch.tensor(0.0)
        criterion = nn.MSELoss(reduction='none')
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, semi_targets, idx = data
                # print(torch.max(labels), torch.min(labels))
                # logger.info("The unique labels are: %s"% str(np.unique(labels.data.cpu().numpy())))
                # logger.info("The unique semi-targets are: %s"% str(np.unique(semi_targets.data.cpu().numpy())))
                # sum_labels += torch.sum(labels > 0)
                # totals += torch.sum(labels > -1)
                # inputs = inputs.to(self.device)
                # semi_targets = semi_targets.to(self.device)
                labels = labels.to(self.device)
                idx = idx.to(self.device)

                inputs, labels, semi_targets, _ = data  # labels was _ previously
                inputs, semi_targets = inputs.to(self.device), semi_targets.to(self.device)
                targets = (semi_targets < 0)
                targets = targets.type(torch.FloatTensor).to(self.device).reshape(-1, 1)
                outputs = ae_net(inputs)
                loss = self.criterion(outputs, targets)

                epoch_loss += loss.item()
                
                n_batches += 1
                scores = outputs
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

                epoch_loss += loss.item()
                n_batches += 1

        # self.psnr = psnr
        if verbose:
            print("label sum: ", sum_labels, len(test_loader), totals)
        self.test_time = time.time() - start_time
        self.test_scores = idx_label_score

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        # print(scores)
        self.test_auc = roc_auc_score(labels, scores)
        
        # Log results
        if verbose:
            logger.info('Test Loss: {:.6f}'.format(epoch_loss / n_batches))
            logger.info('Test AUC: {:.2f}%'.format(100. * self.test_auc))
        # logger.info('Test Time: {:.3f}s'.format(self.test_time))
        # logger.info('Finished testing.')

        return self.test_auc
