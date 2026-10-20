from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from sklearn.metrics import roc_auc_score

import logging
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# import cv2

def log_gg(x, mu, alpha, beta, delta, tau, eps):
    temp = (
            - (((torch.abs(mu - x) / (delta + alpha)) ** 2 + eps) ** (0.5 * (tau + beta)))
            + (torch.log(tau + beta))
            - (torch.lgamma(1 / (tau + beta)))
            - (torch.log(delta + alpha))
            - (torch.log(torch.tensor(2.0)))
    )
    return torch.sum(temp, dim=tuple(range(1, mu.dim())))

class AETrainer(BaseTrainer):

    def __init__(self,eta: float, dataset_name: str = 'cifar10', optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150, lr_milestones: tuple = (),
                 batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda', n_jobs_dataloader: int = 0, recon_param: float = 0.5, eps: float = 0.1, tau: float = 0.1,
                 delta: float = 0.1):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.eps = eps
        self.eta = eta
        self.recon_param = recon_param

        self.tau = tau
        self.delta = delta
        
        
        self.dataset_name = dataset_name

    def train(self, dataset: BaseADDataset, ae_net: BaseNet):
        logger = logging.getLogger()

        # Get train data loader
        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set loss
        criterion = nn.MSELoss(reduction='none')

        # Set device
        ae_net = ae_net.to(self.device)
        criterion = criterion.to(self.device)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(ae_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Training
        logger.info('Starting pretraining...')
        start_time = time.time()
        ae_net.train()

        for epoch in range(self.n_epochs):

            epoch_loss = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                inputs, labels, semi_targets, _ = data # labels was _ previously
                inputs, semi_targets = inputs.to(self.device), semi_targets.to(self.device)
                

                # Zero the network parameter gradients
                optimizer.zero_grad()              

                # Update network parameters via backpropagation: forward + backward + optimize               

                outputs_encoded_mu, outputs_encoded_alpha, outputs_encoded_beta, outputs_recons_mu, outputs_recons_alpha, outputs_recons_beta, sample = ae_net(inputs, ablation_type='B')
                outputs_encoded_alpha.detach()
                outputs_encoded_beta.detach()
                outputs_recons_alpha.detach()
                outputs_recons_beta.detach()

                outputs_encoded_alpha = torch.ones_like(outputs_encoded_alpha)
                outputs_encoded_beta = torch.ones_like(outputs_encoded_beta)*2
                outputs_recons_alpha = torch.ones_like(outputs_recons_alpha)
                outputs_recons_beta = torch.ones_like(outputs_recons_beta)*2
                
                dist_recon = (-1) * log_gg(outputs_recons_mu, inputs, (outputs_recons_alpha), (outputs_recons_beta),
                                           self.delta, self.tau, self.eps)

                losses= torch.where(semi_targets == 0, dist_recon, self.eta * ((dist_recon + self.eps) ** semi_targets.float()))

                kld = log_gg(outputs_encoded_mu, sample, (outputs_encoded_alpha), (outputs_encoded_beta), self.delta, self.tau, self.eps) + 0.5 * torch.norm(sample, dim=1)**2
                
                losses+= self.recon_param * torch.where(semi_targets == 0, kld, self.eta * ((kld + self.eps) ** semi_targets.float()))

                loss = torch.mean(losses)
                loss.backward()
                # optimizer_joint.step()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            print("number of batches=", n_batches)
            logger.info(f'| Step 1: Epoch: {epoch + 1:03}/{self.n_epochs:03} | Train Time: {epoch_train_time:.3f}s '
                        f'| Train Loss: {epoch_loss / n_batches:.6f} |')
        
        for epoch in range(self.n_epochs):

            epoch_loss = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                inputs, labels, semi_targets, _ = data # labels was _ previously
                inputs, semi_targets = inputs.to(self.device), semi_targets.to(self.device)
                

                # Zero the network parameter gradients
                optimizer.zero_grad()              

                # Update network parameters via backpropagation: forward + backward + optimize               

                outputs_encoded_mu, outputs_encoded_alpha, outputs_encoded_beta, outputs_recons_mu, outputs_recons_alpha, outputs_recons_beta, sample = ae_net(inputs, ablation_type='A')
                outputs_encoded_alpha.detach()
                outputs_encoded_beta.detach()
                # outputs_recons_alpha.detach()
                outputs_recons_beta.detach()
                
                dist_recon = (-1) * log_gg(outputs_recons_mu, inputs, torch.exp(outputs_recons_alpha), torch.exp(outputs_recons_beta),
                                           self.delta, self.tau, self.eps)

                losses= torch.where(semi_targets == 0, dist_recon, self.eta * ((dist_recon + self.eps) ** semi_targets.float()))

                kld = log_gg(outputs_encoded_mu, sample, torch.exp(outputs_encoded_alpha), torch.exp(outputs_encoded_beta), self.delta, self.tau, self.eps) + 0.5 * torch.norm(sample, dim=1)**2
                
                losses+= self.recon_param * torch.where(semi_targets == 0, kld, self.eta * ((kld + self.eps) ** semi_targets.float()))

                loss = torch.mean(losses)
                loss.backward()
                # optimizer_joint.step()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            print("number of batches=", n_batches)
            logger.info(f'| Step 2: Epoch: {epoch + 1:03}/{self.n_epochs:03} | Train Time: {epoch_train_time:.3f}s '
                        f'| Train Loss: {epoch_loss / n_batches:.6f} |')

        for epoch in range(self.n_epochs):

            epoch_loss = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                inputs, labels, semi_targets, _ = data # labels was _ previously
                inputs, semi_targets = inputs.to(self.device), semi_targets.to(self.device)
                

                # Zero the network parameter gradients
                optimizer.zero_grad()              

                # Update network parameters via backpropagation: forward + backward + optimize               

                outputs_encoded_mu, outputs_encoded_alpha, outputs_encoded_beta, outputs_recons_mu, outputs_recons_alpha, outputs_recons_beta, sample = ae_net(inputs, ablation_type='A')
                outputs_encoded_alpha.detach()
                outputs_encoded_beta.detach()
                # outputs_recons_alpha.detach()
                # outputs_recons_beta.detach()

                
                dist_recon = (-1) * log_gg(outputs_recons_mu, inputs, torch.exp(outputs_recons_alpha), torch.exp(outputs_recons_beta),
                                           self.delta, self.tau, self.eps)

                losses= torch.where(semi_targets == 0, dist_recon, self.eta * ((dist_recon + self.eps) ** semi_targets.float()))

                kld = log_gg(outputs_encoded_mu, sample, torch.exp(outputs_encoded_alpha), torch.exp(outputs_encoded_beta), self.delta, self.tau, self.eps) + 0.5 * torch.norm(sample, dim=1)**2
                
                losses+= self.recon_param * torch.where(semi_targets == 0, kld, self.eta * ((kld + self.eps) ** semi_targets.float()))

                loss = torch.mean(losses)
                loss.backward()
                # optimizer_joint.step()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            print("number of batches=", n_batches)
            logger.info(f'| Step 3: Epoch: {epoch + 1:03}/{self.n_epochs:03} | Train Time: {epoch_train_time:.3f}s '
                        f'| Train Loss: {epoch_loss / n_batches:.6f} |')

        for epoch in range(self.n_epochs):

            epoch_loss = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                inputs, labels, semi_targets, _ = data # labels was _ previously
                inputs, semi_targets = inputs.to(self.device), semi_targets.to(self.device)
                

                # Zero the network parameter gradients
                optimizer.zero_grad()              

                # Update network parameters via backpropagation: forward + backward + optimize               

                outputs_encoded_mu, outputs_encoded_alpha, outputs_encoded_beta, outputs_recons_mu, outputs_recons_alpha, outputs_recons_beta, sample = ae_net(inputs, ablation_type='A')
                # outputs_encoded_alpha.detach()
                outputs_encoded_beta.detach()
                # outputs_recons_alpha.detach()
                # outputs_recons_beta.detach()

                                
                dist_recon = (-1) * log_gg(outputs_recons_mu, inputs, torch.exp(outputs_recons_alpha), torch.exp(outputs_recons_beta),
                                           self.delta, self.tau, self.eps)

                losses= torch.where(semi_targets == 0, dist_recon, self.eta * ((dist_recon + self.eps) ** semi_targets.float()))

                kld = log_gg(outputs_encoded_mu, sample, torch.exp(outputs_encoded_alpha), torch.exp(outputs_encoded_beta), self.delta, self.tau, self.eps) + 0.5 * torch.norm(sample, dim=1)**2
                
                losses+= self.recon_param * torch.where(semi_targets == 0, kld, self.eta * ((kld + self.eps) ** semi_targets.float()))

                loss = torch.mean(losses)
                loss.backward()
                # optimizer_joint.step()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            print("number of batches=", n_batches)
            logger.info(f'| Step 4: Epoch: {epoch + 1:03}/{self.n_epochs:03} | Train Time: {epoch_train_time:.3f}s '
                        f'| Train Loss: {epoch_loss / n_batches:.6f} |')
        
        for epoch in range(self.n_epochs):

            epoch_loss = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                inputs, labels, semi_targets, _ = data # labels was _ previously
                inputs, semi_targets = inputs.to(self.device), semi_targets.to(self.device)
                

                # Zero the network parameter gradients
                optimizer.zero_grad()              

                # Update network parameters via backpropagation: forward + backward + optimize               

                outputs_encoded_mu, outputs_encoded_alpha, outputs_encoded_beta, outputs_recons_mu, outputs_recons_alpha, outputs_recons_beta, sample = ae_net(inputs, ablation_type='A')
                # outputs_encoded_alpha.detach()
                # outputs_encoded_beta.detach()
                # outputs_recons_alpha.detach()
                # outputs_recons_beta.detach()
                
                dist_recon = (-1) * log_gg(outputs_recons_mu, inputs, torch.exp(outputs_recons_alpha), torch.exp(outputs_recons_beta),
                                           self.delta, self.tau, self.eps)

                losses= torch.where(semi_targets == 0, dist_recon, self.eta * ((dist_recon + self.eps) ** semi_targets.float()))

                kld = log_gg(outputs_encoded_mu, sample, torch.exp(outputs_encoded_alpha), torch.exp(outputs_encoded_beta), self.delta, self.tau, self.eps) + 0.5 * torch.norm(sample, dim=1)**2
                
                losses+= self.recon_param * torch.where(semi_targets == 0, kld, self.eta * ((kld + self.eps) ** semi_targets.float()))

                loss = torch.mean(losses)
                loss.backward()
                # optimizer_joint.step()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            print("number of batches=", n_batches)
            logger.info(f'| Step 5: Epoch: {epoch + 1:03}/{self.n_epochs:03} | Train Time: {epoch_train_time:.3f}s '
                        f'| Train Loss: {epoch_loss / n_batches:.6f} |')

        """
        for epoch in range(self.n_epochs):

            scheduler.step()
            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            epoch_loss = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                inputs, _, _, _ = data
                inputs = inputs.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize

                # _, _, _, rec, _, _, sample = ae_net(inputs)

                outputs_encoded_mu, outputs_encoded_alpha, outputs_encoded_beta, outputs_recons_mu, outputs_recons_alpha, outputs_recons_beta, sample = ae_net(inputs)
                
                outputs_encoded_alpha.detach()
                outputs_encoded_beta.detach()
                outputs_recons_alpha.detach()
                outputs_recons_beta.detach()

                outputs_encoded_alpha = torch.log(torch.ones_like(outputs_encoded_alpha))
                outputs_encoded_beta = torch.log(torch.ones_like(outputs_encoded_beta)*2.0)

                outputs_recons_alpha = torch.log(torch.ones_like(outputs_recons_alpha))
                outputs_recons_beta = torch.log(torch.ones_like(outputs_recons_beta)*2.0)


                rec_loss = criterion(outputs_recons_mu, inputs)
                
                sample_loss = 1e-2 * (torch.norm(sample, dim=1) ** 2)
                

                loss = torch.mean(rec_loss) + torch.mean(sample_loss)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info(f'| Epoch: {epoch + 1:03}/{self.n_epochs:03} | Train Time: {epoch_train_time:.3f}s '
                        f'| Train Loss: {epoch_loss / n_batches:.6f} |')
        """
        self.train_time = time.time() - start_time
        logger.info('Pretraining Time: {:.3f}s'.format(self.train_time))
        logger.info('Finished pretraining.')

        return ae_net

    def test(self, dataset: BaseADDataset, ae_net: BaseNet, xp_path: str):
        logger = logging.getLogger()

        # Get test data loader
        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set loss
        criterion = nn.MSELoss(reduction='none')

        # Set device for network
        ae_net = ae_net.to(self.device)
        criterion = criterion.to(self.device)

        # Testing
        logger.info('Testing autoencoder...')
        epoch_loss = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        ae_net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, _, idx = data
                inputs, labels, idx = inputs.to(self.device), labels.to(self.device), idx.to(self.device)

                # _, rec = ae_net(inputs)    
                # _, _, _, rec, _, _, sample = ae_net(inputs)
                outputs_encoded_mu, outputs_encoded_alpha, outputs_encoded_beta, outputs_recons_mu, outputs_recons_alpha, outputs_recons_beta, sample = ae_net(inputs)

                rec_loss = criterion(outputs_recons_mu, inputs)
                sample_loss = (torch.norm(sample, dim=1)**2)
                # rec_loss += sample_loss
                # rec_loss = (criterion(rec, inputs) + self.eps) ** (self.p/2)
                scores = torch.mean(rec_loss, dim=tuple(range(1, outputs_recons_mu.dim()))) + torch.mean(sample_loss)

                # Save triple of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

                loss = torch.mean(rec_loss)
                epoch_loss += loss.item()
                n_batches += 1

        self.test_time = time.time() - start_time

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        self.test_auc = roc_auc_score(labels, scores)

        # Log results
        logger.info('Test Loss: {:.6f}'.format(epoch_loss / n_batches))
        logger.info('Test AUC: {:.2f}%'.format(100. * self.test_auc))
        logger.info('Test Time: {:.3f}s'.format(self.test_time))
        logger.info('Finished testing autoencoder.')
