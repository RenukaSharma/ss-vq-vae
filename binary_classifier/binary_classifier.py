import json
import torch

from base.base_dataset import BaseADDataset
from networks_.main import build_autoencoder
from optim.DeepSAD_trainer import DeepSADTrainer
from optim.ae_trainer import AETrainer
from optim.BC_trainer import BCTrainer


class BC(object):
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

        self.xp_path = xp_path

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

    def set_network(self, net_name):
        """Builds the neural network phi."""
        self.net_name = net_name
        # self.net, self.dec_net = build_network(net_name)
        

    def train(self, dataset: BaseADDataset, dataset_name: str = 'cifar10',optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 50,
              lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
              n_jobs_dataloader: int = 0 , eps: float = 0.1, tau: float =0.1, delta: float=0.1, const: float=0.1, ablation_type='A'):
        """Trains the Deep SAD model on the training data."""
        if self.ae_net is None:
            self.ae_net = build_autoencoder(self.net_name)
        self.optimizer_name = optimizer_name
        self.trainer = BCTrainer(self.c, self.eta, dataset_name= dataset_name ,optimizer_name=optimizer_name, lr=lr, n_epochs=n_epochs,
                                      lr_milestones=lr_milestones, batch_size=batch_size, weight_decay=weight_decay,
                                      device=device, n_jobs_dataloader=n_jobs_dataloader, recon_param = self.recon_param, eps = eps, tau= tau, delta = delta, const = const, xp_path=self.xp_path)
        # Get the model
        self.net, self.dec_net, self.ae_net = self.trainer.train(dataset, self.net, self.dec_net, self.ae_net, ablation_type)
        self.results['train_time'] = self.trainer.train_time
        # self.c = self.trainer.c.cpu().data.numpy().tolist()  # get as list

    def test(self, dataset: BaseADDataset, device: str = 'cuda', n_jobs_dataloader: int = 0, xp_path: str ='../'):
        """Tests the Deep SAD model on the test data."""

        # if self.trainer is None:
        #     self.trainer = DeepSADTrainer(self.c, self.eta, device=device, n_jobs_dataloader=n_jobs_dataloader)

        self.trainer.test(dataset, self.net, self.dec_net, self.ae_net, xp_path = xp_path)

        # Get results
        self.results['test_auc'] = self.trainer.test_auc
        self.results['test_time'] = self.trainer.test_time
        self.results['psnr'] = self.trainer.psnr
        
        self.results['test_scores'] = self.trainer.test_scores
        return self.trainer.test_auc
    
    def pretrain(self, dataset: BaseADDataset, dataset_name: str = 'cifar10', optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 100,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0, xp_path: str ='../'):
        """Pretrains the weights for the Deep SAD network phi via autoencoder."""

        # Set autoencoder network
        self.ae_net = build_autoencoder(self.net_name)
        print("Net nae: ", self.net_name)
        # Train
        self.ae_optimizer_name = optimizer_name
        self.ae_trainer = AETrainer(dataset_name= dataset_name ,optimizer_name=optimizer_name, lr=lr, n_epochs=n_epochs, lr_milestones=lr_milestones,
                                    batch_size=batch_size, weight_decay=weight_decay, device=device,
                                    n_jobs_dataloader=n_jobs_dataloader)
        self.ae_net = self.ae_trainer.train(dataset, self.ae_net)

        # Get train results
        self.ae_results['train_time'] = self.ae_trainer.train_time

        # Test
        self.ae_trainer.test(dataset, self.ae_net, xp_path = xp_path)

        # Get test results
        self.ae_results['test_auc'] = self.ae_trainer.test_auc
        self.ae_results['test_time'] = self.ae_trainer.test_time

        # Initialize Deep SAD network weights from pre-trained encoder
        self.init_network_weights_from_pretraining()
    

    
    def init_network_weights_from_pretraining(self):
        """Initialize the Deep SAD network weights from the encoder weights of the pretraining autoencoder."""

        net_dict = self.net.state_dict()
        ae_net_dict = self.ae_net.state_dict()

        # Filter out decoder network keys
        ae_net_dict = {k: v for k, v in ae_net_dict.items() if (k in net_dict)}
        # Overwrite values in the existing state_dict
        net_dict.update(ae_net_dict)
        # Load the new state_dict
        self.net.load_state_dict(net_dict)

    def save_model(self, export_model, save_ae=True):
        """Save Deep SAD model to export_model."""
        
        net_dict = self.net.state_dict() if self.net else None
        print("self.net is None?:", self.net is None)
        ae_net_dict = self.ae_net.state_dict() if save_ae else None
        print("self.ae_net is None?", self.ae_net is None)

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
