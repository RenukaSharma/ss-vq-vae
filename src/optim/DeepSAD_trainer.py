from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve

import logging
import time
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
import pylab as pl
import copy
import sys
from PIL import Image
import matplotlib.pyplot as plt

import torch.nn.functional as F
from scipy.signal import savgol_filter
# import cv2


def log_gg(x, mu,  beta, tau, eps):
    # temp = ((-1) * (torch.abs(mu - x))**(beta+tau)) - torch.log(torch.tensor(2.0)) - torch.log( torch.lgamma(1 + 1.0/(beta+tau)))

    temp = (
            - (((torch.abs(mu - x) ) ** 2 + eps) ** (0.5 * (tau + beta)))
            + (torch.log(tau + beta))
            - (torch.lgamma(1 / (tau + beta)))
            # - (torch.log( alpha))
            - (torch.log(torch.tensor(2.0)))
    )
    # print("torch.abs(x)=", torch.abs(x))
    # print("beta+tau=", beta+tau)
    # print("(2 * torch.lgamma(1 + 1.0/(beta+tau)))=", (2 * torch.lgamma(1 + 1.0/(beta+tau))))
    # print("The value for temp is", temp)
    return torch.sum(temp, dim=tuple(range(1, mu.dim())))
"""
def log_gg(x, mu, beta, tau, eps):
    alpha = torch.ones_like(beta)
    delta = tau
    # print("shape of x:", x.shape)
    # print("shape of beta:", beta.shape)
    temp = (
            - (((torch.abs(mu - x) / ( alpha)) ** 2 + eps) ** (0.5 * (tau + beta)))
            + (torch.log(tau + beta))
            - (torch.lgamma(1 / (tau + beta)))
            # - (torch.log( alpha))
            # - (torch.log(torch.tensor(2.0)))
    )
    return torch.sum(temp, dim=tuple(range(1, mu.dim())))
"""
"""
def log_gg(x, mu, alpha, beta, delta, tau, eps):
    temp = (
            - (((torch.abs(mu - x) / ( alpha)) ** 2 + eps) ** (0.5 * (tau + beta)))
            + (torch.log(tau + beta))
            - (torch.lgamma(1 / (tau + beta)))
            - (torch.log( alpha))
            - (torch.log(torch.tensor(2.0)))
    )
    return torch.sum(temp, dim=tuple(range(1, mu.dim())))
"""

class DeepSADTrainer(BaseTrainer):

    def __init__(self, c, eta: float, dataset_name: str = 'cifar10', optimizer_name: str = 'adam', lr: float = 0.001,
                 n_epochs: int = 150,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 0.99, 
                 device: str = 'cuda',
                 n_jobs_dataloader: int = 0, recon_param: float = 0.5, latent_param: float = 0.5, eps: float = 1e-5, tau: float = 0.1,
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

        # self.batch_size = 128

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
        # self.recon_param = recon_param
        # self.latent_param = latent_param

        # small real-values
        # self.tau = tau
        # self.delta = delta
        # self.const = const



    def train(self, dataset: BaseADDataset = 'cifar10', net: BaseNet = None, dec_net: BaseNet = None,
              ae_net: BaseNet = None, ablation_type='A'):

        logger = logging.getLogger()
        
        ###Best model vals: START:1/3################################
        best_score = -np.inf
        best_model = None
        ###Best model vals: END:1/3################################
        

        # Get train data loader
        train_loader, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)
        # copy_dataset = copy.deepcopy(dataset)
        # Set device for network        

        ae_net = ae_net.to(self.device)

        # Set optimizer (Adam optimizer for now)    

        optimizer = optim.Adam(ae_net.parameters(), lr=self.lr, weight_decay=self.weight_decay, amsgrad=False)

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)
        self.criterion = nn.MSELoss(reduction='none')
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

        self.train_res_recon_error = []
        self.train_res_perplexity = []
        ae_net.train()
        
        for epoch in range(self.n_epochs):
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
                optimizer.zero_grad()
                data_variance=0.0
                # import ipdb; ipdb.set_trace()                
                for i in range(len(data)):
                    data_variance += np.var(data[i].numpy() / 255.0)
                if ablation_type == 'A':
                    # print("#####$$$$$$$Data variance=", data_variance)
                    losses, recon_error_semisup, perplexity, data_recon, z, quantized = self.loss_A(data, ae_net)
                    # import ipdb; ipdb.set_trace()
                    self.train_res_recon_error.append(recon_error_semisup.item())
                    # self.train_res_recon_error.append([t.numpy() for t in recon_error_semisup])
                    self.train_res_perplexity.append(perplexity.item())
                elif ablation_type == 'D':
                    losses, latent, dist_recon, reco_loss, _, _, _ = self.loss_D(data, ae_net)
                elif ablation_type == 'E':
                    losses, latent, dist_recon, reco_loss, _, _, _ = self.loss_E(data, ae_net)
                elif ablation_type == 'VAE':
                    losses, latent, dist_recon, reco_loss, _, _, _ = self.loss_VAE(data, ae_net)               

                # loss = torch.mean(losses)
                loss = losses
                loss.backward()
                # optimizer_joint.step()
                optimizer.step()
                epoch_loss += loss.item()
                # kl_loss += torch.mean(latent).item()
                # image_loss += torch.mean(dist_recon).item()
                # recon_loss += torch.mean(reco_loss).item()
                n_batches += 1

                if (epoch+1) % 100 == 0:
                    print('%d iterations' % (epoch+1))
                    print('recon_error: %.3f' % np.mean(self.train_res_recon_error[-100:]))
                    print('perplexity: %.3f' % np.mean(self.train_res_perplexity[-100:]))
                    print()

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            test_auc = self.test(dataset, ae_net, '../', verbose=False)
            
            ###Best model vals: START:2/3################################
            if test_auc > best_score:
                best_score = test_auc
                best_model = copy.deepcopy(ae_net)
            ###Best model vals: END:2/3###################################
            
            # if(epoch%100 != 0):
            #     test_auc = self.test(dataset, net, dec_net, ae_net, '../', verbose=False)
            # else:
            #     test_auc = self.test(dataset, net, dec_net, ae_net, '../', verbose=True)
            # test_auc = 0
            logger.info(f'| Epoch: {epoch + 1:03}/{self.n_epochs:03} | Test AUC: {test_auc:.3f} '
                        f'| Train Loss: {epoch_loss / n_batches:.6f}')

            # logger.info(f'| Epoch: {epoch + 1:03}/{self.n_epochs:03} | Test AUC: {test_auc:.3f} '
            #             f'| Train Loss: {epoch_loss / n_batches:.6f} | Recon loss: {recon_loss / n_batches}'
            #             f'| Latent loss: {kl_loss / n_batches}       | Image loss: {image_loss / n_batches}')
            
            self.test_aucs.append(test_auc)
            self.losses.append(epoch_loss / n_batches)
            # self.latent_losses.append(kl_loss / n_batches)
            # self.image_losses.append(image_loss / n_batches)
            # self.recon_losses.append(recon_loss / n_batches)
        
        
        ###Best model vals: START:3/3################################
        ae_net = copy.deepcopy(best_model)
        ###Best model vals: END:3/3###################################
        
        
        self.train_time = time.time() - start_time
        logger.info('Training Time: {:.3f}s'.format(self.train_time))
        logger.info('Finished training.')

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot(self.test_aucs, label="test_auc")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(self.losses, label="loss")
        # plt.plot(self.image_losses, label="image loss")
        # plt.plot(self.latent_losses, label="latent loss")
        plt.legend()
        plt.savefig(self.xp_path + "/plots.png")

        #### for VQ-VAE: START ##########
        # train_res_recon_error_smooth = savgol_filter(self.train_res_recon_error, 201, 7)
        # train_res_perplexity_smooth = savgol_filter(self.train_res_perplexity, 201, 7)

        # f = plt.figure(figsize=(16,8))
        # ax = f.add_subplot(1,2,1)
        # ax.plot(train_res_recon_error_smooth)
        # ax.set_yscale('log')
        # ax.set_title('Smoothed NMSE.')
        # ax.set_xlabel('iteration')

        # ax = f.add_subplot(1,2,2)
        # ax.plot(train_res_perplexity_smooth)
        # ax.set_title('Smoothed Average codebook usage (perplexity).')
        # ax.set_xlabel('iteration')
        #### for VQ-VAE: END ############

        return net, dec_net, ae_net

    def loss_A(self, data, ae_net):
        inputs, labels, semi_targets, _ = data  # labels was _ previously
        inputs_dv = inputs
        # import ipdb; ipdb.set_trace()
        # data_variance = np.var(inputs_dv.numpy() / 255.0) # removing later ; ; Jul 2, 2022
        inputs, semi_targets = inputs.to(self.device), semi_targets.to(self.device)

        vq_loss, data_recon, z, quantized, perplexity = ae_net(inputs)
        # import ipdb; ipdb.set_trace()
        vq_loss_semisup = torch.where(semi_targets == -1, self.eta * (1- (vq_loss + self.eps)), vq_loss)
        # vq_loss_semisup = torch.where(semi_targets == 0, vq_loss, self.eta * ((vq_loss + self.eps) * semi_targets.float())) 
        # vq_loss_semisup = torch.where(semi_targets == 0, vq_loss, self.eta * ((vq_loss + self.eps) ** semi_targets.float())) # works better
        # import ipdb; ipdb.set_trace()
        # recon_error = F.mse_loss(data_recon, inputs) / data_variance  # removing data_variance ; Jul 2, 2022
        # recon_error = F.mse_loss(data_recon, inputs)  
        recon_error = F.l1_loss(data_recon, inputs) # removing data_variance and changing  to l1 loss ; Jul 2, 2022
        # import ipdb; ipdb.set_trace()
        recon_error_semisup = torch.where(semi_targets == -1, self.eta * (1 - recon_error + self.eps), recon_error)
        # recon_error_semisup = torch.where(semi_targets == 0, recon_error, self.eta * ((recon_error + self.eps) * semi_targets.float()))
        # recon_error_semisup = torch.where(semi_targets == 0, recon_error, self.eta * ((recon_error + self.eps) ** semi_targets.float())) # works better
        losses = recon_error_semisup + vq_loss_semisup
        
        loss = torch.mean(losses)
        recon_error_semisup_mean = torch.mean(recon_error_semisup)
        return loss, recon_error_semisup_mean, perplexity, data_recon, z, quantized
        
        # outputs_encoded_mu, outputs_encoded_beta, outputs_recons_mu, outputs_recons_beta, sample = ae_net(inputs, ablation_type='A')
        
        # dist_first = log_gg(sample, outputs_encoded_mu,
        #                     torch.exp(outputs_encoded_beta), self.tau, self.eps)
        # dist_second = 0.5 * (torch.norm(sample, dim=1) ** 2)
        # dist_recon = self.recon_param * (-1) * log_gg(inputs, outputs_recons_mu, 
        #                            torch.exp(outputs_recons_beta),
        #                            self.tau, self.eps)

        # dist = dist_first + dist_second + dist_recon
        # latent = self.latent_param*(dist_first + dist_second) 
        # recon_semisup = torch.where(semi_targets == 0, dist_recon, self.eta * ((dist_recon + self.eps) ** semi_targets.float()))
        # latent_semisup = torch.where(semi_targets == 0, latent, self.eta * ((latent + self.eps) ** semi_targets.float()))                
        # losses = latent_semisup + recon_semisup
        # # losses = torch.where(semi_targets == 0, dist, self.eta * ((dist + self.eps) ** semi_targets.float()))
        # recon_loss = self.criterion(inputs, outputs_recons_mu)
        # losses = torch.mean(losses)
        # return losses, latent_semisup, dist_recon, recon_semisup, outputs_encoded_mu, outputs_recons_mu, sample

    def loss_D(self, data, ae_net):
        inputs, labels, semi_targets, _ = data  # labels was _ previously
        inputs, semi_targets = inputs.to(self.device), semi_targets.to(self.device)

        outputs_encoded_mu, outputs_encoded_beta, outputs_recons_mu, outputs_recons_beta, sample = ae_net(
            inputs, ablation_type='D')
        
        # outputs_encoded_alpha.detach()
        outputs_encoded_beta.detach()
        # outputs_recons_alpha.detach()
        outputs_recons_beta.detach()

        # outputs_encoded_alpha = torch.ones_like(outputs_encoded_mu) * 1.0
        outputs_encoded_beta = torch.ones_like(outputs_encoded_mu) * 2.0

        # outputs_recons_alpha = torch.ones_like(outputs_recons_alpha) *1.0
        outputs_recons_beta = torch.ones_like(outputs_recons_beta) * 2.0
        
        image_space = self.recon_param * torch.mean((inputs - outputs_recons_mu) ** 2, dim=(1, 2, 3))
        
        recon_loss = torch.mean(self.criterion(inputs, outputs_recons_mu),  dim=(1, 2, 3))
        # losses = recon_loss.copy()

        # losses = torch.where(semi_targets == 0, image_space, self.eta * ((image_space + self.eps) ** semi_targets.float()))
        losses = torch.where(semi_targets == 0, recon_loss, self.eta * ((recon_loss + self.eps) * semi_targets.float()))

        losses = torch.mean(losses)

        return losses, torch.zeros_like(image_space), image_space, image_space, outputs_encoded_mu, outputs_recons_mu, sample


    def loss_E(self, data, ae_net):
        inputs, labels, semi_targets, _ = data  # labels was _ previously
        inputs, semi_targets = inputs.to(self.device), semi_targets.to(self.device)

        outputs_encoded_mu, outputs_encoded_beta, outputs_recons_mu, outputs_recons_beta, sample = ae_net(
            inputs, ablation_type='E')
        
        # outputs_encoded_alpha.detach()
        outputs_encoded_beta.detach()
        # outputs_recons_alpha.detach()
        outputs_recons_beta.detach()

        # outputs_encoded_alpha = torch.ones_like(outputs_encoded_mu) * 1.0
        outputs_encoded_beta = torch.ones_like(outputs_encoded_mu) * 2.0

        # outputs_recons_alpha = torch.ones_like(outputs_recons_alpha) *1.0
        outputs_recons_beta = torch.ones_like(outputs_recons_beta) * 2.0
        
        latent = self.latent_param * torch.norm(outputs_encoded_mu, dim=1)**2
        # latent = self.recon_param*torch.sum(sample**2, dim=1)

        image_space = torch.mean((inputs - outputs_recons_mu) ** 2, dim=(1, 2, 3))
        latent_semisup = torch.where(semi_targets == 0, latent, self.eta * ((latent + self.eps) ** semi_targets.float()))
        # latent_semisup = torch.where(semi_targets == 0, latent, self.eta * ((latent + self.eps) * semi_targets.float()))
        # recon_semisup = torch.where(semi_targets == 0, image_space, self.eta * ((image_space + self.eps) ** semi_targets.float()))

        # losses = recon_semisup + latent_semisup
        losses = latent_semisup

        recon_loss = self.criterion(inputs, outputs_recons_mu)
        # losses = torch.where(semi_targets == 0, losses, self.eta * ((losses + self.eps) ** semi_targets.float()))
        losses = torch.mean(losses)
        return losses, latent, torch.zeros_like(image_space), recon_loss, outputs_encoded_mu, outputs_recons_mu, sample


    def loss_VAE(self, data, ae_net):
        """
        Gaussian Variational Autoencoder loss
        """
        # import ipdb; ipdb.set_trace()
        def kl_g(mu, log_sigma):
            return 0.5 * (mu ** 2 + torch.exp(log_sigma) ** 2 - 1) - log_sigma

        inputs, labels, semi_targets, _ = data  # labels was _ previously
        inputs, semi_targets = inputs.to(self.device), semi_targets.to(self.device)

        outputs_encoded_mu, outputs_encoded_alpha, outputs_encoded_beta, outputs_recons_mu, outputs_recons_alpha, outputs_recons_beta, sample = ae_net(
            inputs, ablation_type='VAE')

        kl_loss = kl_g(outputs_encoded_mu, outputs_encoded_alpha)
        kl_loss = self.latent_param * torch.sum(kl_loss, dim=1)
        standard_recons_beta = torch.ones_like(outputs_recons_mu) * 2.0
        # dist_recon = (-1) * log_gg(inputs, outputs_recons_mu, torch.exp(outputs_recons_alpha),
                                #    standard_recons_beta,
                                #    self.delta, 0.0, self.eps)

        kl_semisup = torch.where(semi_targets == 0, kl_loss, self.eta * ((kl_loss + self.eps) ** semi_targets.float()))
        # kl_semisup = torch.where(semi_targets == 0, kl_loss, self.eta * ((kl_loss + self.eps) * semi_targets.float()))
        

        # losses = kl_semisup + recon_semisup
        recon_loss = self.criterion(inputs, outputs_recons_mu)
        recon_loss = self.recon_param * torch.mean(recon_loss, dim=(1, 2, 3))

        recon_semisup = torch.where(semi_targets == 0, recon_loss, self.eta * ((recon_loss + self.eps) ** semi_targets.float()))
        # recon_semisup = torch.where(semi_targets == 0, recon_loss, self.eta * ((recon_loss + self.eps) * semi_targets.float()))
        
        losses = kl_semisup + recon_semisup
        losses = torch.mean(losses)
        # losses = recon_loss
        # losses = torch.where(semi_targets == 0, losses, self.eta * ((losses + self.eps) ** semi_targets.float()))
        return losses, kl_loss, recon_loss, recon_loss, outputs_encoded_mu, outputs_recons_mu, sample


    def loss_test(self, data, ae_net):
        inputs, labels, semi_targets, _ = data  # labels was _ previously
        inputs, semi_targets = inputs.to(self.device), semi_targets.to(self.device)

        outputs_encoded_mu, outputs_encoded_alpha, outputs_encoded_beta, outputs_recons_mu, outputs_recons_alpha, outputs_recons_beta, sample = ae_net(
            inputs, ablation_type='test')

        # dist_first = log_gg(sample, outputs_encoded_mu, torch.exp(outputs_encoded_alpha),
        #                     torch.exp(outputs_encoded_beta), self.delta, self.tau, self.eps)
        dist_second = 0.5 * (torch.norm(sample, dim=1) ** 2)
        dist_recon = (-1) * log_gg(inputs, outputs_recons_mu, torch.exp(outputs_recons_alpha),
                                   torch.exp(outputs_recons_beta),
                                   self.delta, self.tau, self.eps)

        dist = dist_second + dist_recon
        latent = dist_second
        latent_semisup = torch.where(semi_targets == 0, latent, self.eta * ((latent + self.eps) ** semi_targets.float()))
        # latent_semisup = torch.where(semi_targets == 0, latent, self.eta * ((latent + self.eps) * semi_targets.float()))
        recon_semisup = torch.where(semi_targets == 0, dist_recon, self.eta * ((dist_recon + self.eps) ** semi_targets.float()))
        # recon_semisup = torch.where(semi_targets == 0, dist_recon, self.eta * ((dist_recon + self.eps) * semi_targets.float()))

        # print("Beta: ", torch.mean(torch.exp(outputs_encoded_beta)))
        # print(torch.mean(dist_first), torch.mean(dist_second), torch.mean(dist_recon))
        losses = latent_semisup + recon_semisup
        # losses = torch.where(semi_targets == 0, dist, self.eta * ((dist + self.eps) ** semi_targets.float()))
        recon_loss = self.criterion(inputs, outputs_recons_mu)
        return losses, latent, dist_recon, recon_loss, outputs_encoded_mu, outputs_recons_mu, sample


    def test(self, dataset: BaseADDataset = 'cifar10', ae_net: BaseNet = None, xp_path: str = '../', verbose: bool = True):

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

                if self.ablation_type == 'A':
                    # print("#####$$$$$$$Data variance=", data_variance)                    
                    # losses, x_recon, z, quantized, _ = self.loss_A(data, ae_net)     
                    losses, recon_error_semisup, perplexity, data_recon, z, quantized = self.loss_A(data, ae_net)              
                elif self.ablation_type == 'D':
                    losses, latent, dist_recon, reco_loss, outputs_encoded_mu, outputs_recons_mu, sample = self.loss_D(data, ae_net)
                elif self.ablation_type == 'E':
                    losses, latent, dist_recon, reco_loss, outputs_encoded_mu, outputs_recons_mu, sample = self.loss_E(data, ae_net)
                elif self.ablation_type == 'VAE':
                    losses, latent, dist_recon, reco_loss, outputs_encoded_mu, outputs_recons_mu, sample = self.loss_VAE(data, ae_net)
                elif self.ablation_type == 'test':
                    losses, latent, dist_recon, reco_loss, outputs_encoded_mu, outputs_recons_mu, sample = self.loss_VAE(data, ae_net)

                loss = torch.mean(losses)
                # recon = criterion(inputs, outputs_recons_mu)
                # loss = torch.mean(recon)
                # scores = torch.sum((outputs_recons_mu - inputs) ** 2.0, dim=tuple(range(1, outputs_recons_mu.dim())) )
                
                # scores = latent + dist_recon
                # scores = dist_recon
                
                # scores = torch.norm(sample, dim=1) ** 2
                # scores = torch.norm(outputs_encoded_mu, dim=1) ** 2

                # scores = torch.norm(z-quantized)**2
                # import ipdb; ipdb.set_trace()
                scores = torch.norm(z-quantized, dim=1)
                score = torch.norm(scores, dim=(1,2))

                # scores = torch.sum(recon, dim=tuple(range(1, inputs.dim())))
                # scores = torch.norm((inputs - outputs_recons_mu).view(128, -1), dim=1)
                # scores = (1-self.recon_param) * dist + self.recon_param * dist_recon
                # scores = dist + self.recon_param * dist_recon
                # Save triples of (idx, label, score) in a list
                # import ipdb; ipdb.set_trace()
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            score.cpu().data.numpy().tolist()))

                epoch_loss += loss.item()
                n_batches += 1

        # self.psnr = psnr
        # if verbose:
        #     print("label sum: ", sum_labels, len(test_loader), totals)
        self.test_time = time.time() - start_time
        self.test_scores = idx_label_score

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        
        # print("The scores are:", scores)
        # print("The labels are:", labels)

        ## adding for MVTec's object class, especially bottle category, reduced batch size to 8 (from 256, 64, ... - these did'nt work, genertaing the exception):
        try:
            self.test_auc = roc_auc_score(labels, scores)
        except ValueError:
            logger.info("Avoiding the roc_auc_score - Only one class present in y_true error, setting test_auc to -np.inf")
            self.test_auc = -np.inf
        # self.test_auc = roc_auc_score(labels, scores)

        fpr, tpr, thresholds =roc_curve(labels, scores)
        # roc_auc = auc(fpr, tpr)
        print("Area under the ROC curve : %f" % self.test_auc)

        ####################################
        # The optimal cut off would be where tpr is high and fpr is low
        # tpr - (1-fpr) is zero or near to zero is the optimal cut off point
        ####################################
        i = np.arange(len(tpr)) # index for df
        roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
        roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
        # print("optimal threshold",float(list(roc_t['thresholds'])))
        # print("Its type is", type(roc_t['thresholds']))
        
        logger.info('Optimal thershold score: %g' % float((roc_t['thresholds'])))
        logger.info('Stats for scores: mean: %g ' % float(scores.mean()))
        logger.info('Stats for scores: max:  %g' % float(scores.max()))
        logger.info('Stats for scores: min: %g' % float(scores.min()))

        # Plot tpr vs 1-fpr
        if verbose:
            fig, ax = pl.subplots()
            pl.plot(roc['tpr'])
            pl.plot(roc['1-fpr'], color = 'red')
            pl.xlabel('1-False Positive Rate')
            pl.ylabel('True Positive Rate')
            pl.title('Receiver operating characteristic')
            ax.set_xticklabels([])
            pl.savefig(self.xp_path + "/roc_auc_plot.png")

        # for name, param in net.named_parameters():
        #     if name == "p":
        #         net_p_value = param.data
        #     if name == "c":
        #         net_c_value = param.data

        # for name, param in net.named_parameters():
        #     if name == "p":
        #         ae_net_p_value = param.data
        #     if name == "c":
        #         ae_net_c_value = param.data
        # Log results
        if verbose:
            logger.info('Test Loss: {:.6f}'.format(epoch_loss / n_batches))
            logger.info('Test AUC: {:.2f}%'.format(100. * self.test_auc))
        # logger.info('Test Time: {:.3f}s'.format(self.test_time))
        # logger.info('Finished testing.')

        return self.test_auc
