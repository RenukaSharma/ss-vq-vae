# from .mnist_LeNet import MNIST_LeNet, MNIST_LeNet_Autoencoder, MNIST_LeNet_Decoder
from .mnist_LeNet_vae import MNIST_LeNet, MNIST_LeNet_Autoencoder, MNIST_LeNet_Decoder
# from .mnist_LeNet_GG import MNIST_LeNet, MNIST_LeNet_Autoencoder, MNIST_LeNet_Decoder

from .fmnist_LeNet_vae import FashionMNIST_LeNet, FashionMNIST_LeNet_Autoencoder, FashionMNIST_LeNet_Decoder
# from .fmnist_LeNet_GG import FashionMNIST_LeNet, FashionMNIST_LeNet_Autoencoder, FashionMNIST_LeNet_Decoder

# from .fmnist_LeNet_vae import FashionMNIST_LeNet, FashionMNIST_LeNet_Autoencoder, FashionMNIST_LeNet_Decoder

# from .cifar10_LeNet import CIFAR10_LeNet, CIFAR10_LeNet_Autoencoder, CIFAR10_LeNet_Decoder
# from .cifar10_LeNet_vae import CIFAR10_LeNet, CIFAR10_LeNet_Autoencoder, CIFAR10_LeNet_Decoder
from .cifar10_LeNet_vae_nanofibre import CIFAR10_LeNet_nano, CIFAR10_LeNet_Autoencoder_nano, CIFAR10_LeNet_Decoder_nano
from .nanofibre_vae import Nanofibre_LeNet_Autoencoder
from .cifar10_LeNet_GG import CIFAR10_LeNet, CIFAR10_LeNet_Autoencoder, CIFAR10_LeNet_Decoder

from .mvtec_net_convdeconv import MVTec_net_Autoencoder

from .malaria_net import Malaria_net_Ext
# from .mlp import MLP, MLP_Autoencoder
# from .vae import VariationalAutoencoder
# from .dgm import DeepGenerativeModel, StackedDeepGenerativeModel

from .vq_vae_net import VQ_VAE_Model


def build_network(net_name, num_hiddens, num_residual_layers, num_residual_hiddens, 
                 num_embeddings, embedding_dim, commitment_cost, weight_decay, ae_net=None):
    """Builds the neural network."""

    implemented_networks = ('mnist_LeNet', 'mnist_DGM_M2', 'mnist_DGM_M1M2',
                            'fmnist_LeNet', 'fmnist_DGM_M2', 'fmnist_DGM_M1M2',
                            'cifar10_LeNet', 'cifar10_LeNet_nanofibre', 'cifar10_DGM_M2', 'cifar10_DGM_M1M2', 'malaria_net', 'nanofibre_vae',
                            'arrhythmia_mlp', 'cardio_mlp', 'satellite_mlp', 'satimage-2_mlp', 'shuttle_mlp',
                            'thyroid_mlp',
                            'arrhythmia_DGM_M2', 'cardio_DGM_M2', 'satellite_DGM_M2', 'satimage-2_DGM_M2',
                            'shuttle_DGM_M2', 'thyroid_DGM_M2', 'mvtec_net', 'vq_vae_net')
    print("&*&*&*&*&*&&*&*&*&*&*&*&Net name is:", net_name)
    assert net_name in implemented_networks
    ae_net = None
    net = None
    dec_net =None

    if net_name == 'mnist_LeNet':
        net = MNIST_LeNet()
        dec_net = MNIST_LeNet_Decoder()

    if net_name == 'mnist_DGM_M2':
        net = DeepGenerativeModel([1*28*28, 2, 32, [128, 64]], classifier_net=MNIST_LeNet)

    if net_name == 'mnist_DGM_M1M2':
        net = StackedDeepGenerativeModel([1*28*28, 2, 32, [128, 64]], features=ae_net)

    if net_name == 'fmnist_LeNet':
        net = FashionMNIST_LeNet()
        dec_net = FashionMNIST_LeNet_Decoder()

    if net_name == 'fmnist_DGM_M2':
        net = DeepGenerativeModel([1*28*28, 2, 64, [256, 128]], classifier_net=FashionMNIST_LeNet)

    if net_name == 'fmnist_DGM_M1M2':
        net = StackedDeepGenerativeModel([1*28*28, 2, 64, [256, 128]], features=ae_net)

    if net_name == 'cifar10_LeNet':
        net = CIFAR10_LeNet()
        dec_net = CIFAR10_LeNet_Decoder()

    if net_name == 'cifar10_LeNet_nanofibre':
        net = CIFAR10_LeNet_nano()
        dec_net = CIFAR10_LeNet_Decoder_nano()
    
    # if net_name == 'mvtec_net':
    #     net = CIFAR10_LeNet()
    #     dec_net = CIFAR10_LeNet_Decoder()

    if net_name == 'cifar10_DGM_M2':
        net = DeepGenerativeModel([3*32*32, 2, 128, [512, 256]], classifier_net=CIFAR10_LeNet)

    if net_name == 'cifar10_DGM_M1M2':
        net = StackedDeepGenerativeModel([3*32*32, 2, 128, [512, 256]], features=ae_net)

    if net_name == 'malaria_net':
        net = Malaria_net_Ext() 

    if net_name == 'arrhythmia_mlp':
        net = MLP(x_dim=274, h_dims=[128, 64], rep_dim=32, bias=False)

    if net_name == 'cardio_mlp':
        net = MLP(x_dim=21, h_dims=[32, 16], rep_dim=8, bias=False)

    if net_name == 'satellite_mlp':
        net = MLP(x_dim=36, h_dims=[32, 16], rep_dim=8, bias=False)

    if net_name == 'satimage-2_mlp':
        net = MLP(x_dim=36, h_dims=[32, 16], rep_dim=8, bias=False)

    if net_name == 'shuttle_mlp':
        net = MLP(x_dim=9, h_dims=[32, 16], rep_dim=8, bias=False)

    if net_name == 'thyroid_mlp':
        net = MLP(x_dim=6, h_dims=[32, 16], rep_dim=4, bias=False)

    if net_name == 'arrhythmia_DGM_M2':
        net = DeepGenerativeModel([274, 2, 32, [128, 64]])

    if net_name == 'cardio_DGM_M2':
        net = DeepGenerativeModel([21, 2, 8, [32, 16]])

    if net_name == 'satellite_DGM_M2':
        net = DeepGenerativeModel([36, 2, 8, [32, 16]])

    if net_name == 'satimage-2_DGM_M2':
        net = DeepGenerativeModel([36, 2, 8, [32, 16]])

    if net_name == 'shuttle_DGM_M2':
        net = DeepGenerativeModel([9, 2, 8, [32, 16]])

    if net_name == 'thyroid_DGM_M2':
        net = DeepGenerativeModel([6, 2, 4, [32, 16]])
    
    if net_name == 'vq_vae_net':
        print("*&*&*&*&$%$%^$Building the MVTec's VQ-VAE network...")
        ae_net = VQ_VAE_Model(num_hiddens= num_hiddens, num_residual_hiddens= num_residual_hiddens, num_residual_layers= num_residual_layers, embedding_dim= embedding_dim, num_embeddings= num_embeddings, commitment_cost= commitment_cost, decay=weight_decay)

    return net, dec_net


def build_autoencoder(net_name, num_hiddens, num_residual_layers, num_residual_hiddens, 
                 num_embeddings, embedding_dim, commitment_cost, weight_decay):
    """Builds the corresponding autoencoder network."""
    implemented_networks = ('mnist_LeNet', 'mnist_DGM_M1M2',
                            'fmnist_LeNet', 'fmnist_DGM_M1M2',
                            'cifar10_LeNet', 'cifar10_DGM_M1M2', 'malaria_net','cifar10_LeNet_nanofibre', 'nanofibre_vae',
                            'arrhythmia_mlp', 'cardio_mlp', 'satellite_mlp', 'satimage-2_mlp', 'shuttle_mlp',
                            'thyroid_mlp', 'mvtec_net', 'vq_vae_net')
                            
    print("&*&*&*&*&*&&*&*&*&*&*&*&Net name is:", net_name)
    assert net_name in implemented_networks

    ae_net = None
    
    if net_name == 'vq_vae_net':
        print("*&*&*&*&$%$%^$Building the MVTec's VQ-VAE autoencoder network...")
        ae_net = VQ_VAE_Model(num_hiddens= num_hiddens, num_residual_hiddens= num_residual_hiddens, num_residual_layers= num_residual_layers, embedding_dim= embedding_dim, num_embeddings= num_embeddings, commitment_cost= commitment_cost, decay=weight_decay)
        return ae_net

    if net_name == 'mnist_LeNet':
        ae_net = MNIST_LeNet_Autoencoder()

    if net_name == 'mnist_DGM_M1M2':
        ae_net = VariationalAutoencoder([1*28*28, 32, [128, 64]])

    if net_name == 'fmnist_LeNet':
        ae_net = FashionMNIST_LeNet_Autoencoder()

    if net_name == 'fmnist_DGM_M1M2':
        ae_net = VariationalAutoencoder([1*28*28, 64, [256, 128]])

    if net_name == 'cifar10_LeNet':
        ae_net = CIFAR10_LeNet_Autoencoder()

    if net_name == 'cifar10_LeNet_nanofibre':
        ae_net = CIFAR10_LeNet_Autoencoder_nano()
    
    if net_name == 'nanofibre_vae':
        ae_net = Nanofibre_LeNet_Autoencoder()

    if net_name == 'mvtec_net':
        ae_net = MVTec_net_Autoencoder()

    if net_name == 'cifar10_DGM_M1M2':
        ae_net = VariationalAutoencoder([3*32*32, 128, [512, 256]])
    
    if net_name == 'malaria_net':
        net = Malaria_net_Ext()

    if net_name == 'arrhythmia_mlp':
        ae_net = MLP_Autoencoder(x_dim=274, h_dims=[128, 64], rep_dim=32, bias=False)

    if net_name == 'cardio_mlp':
        ae_net = MLP_Autoencoder(x_dim=21, h_dims=[32, 16], rep_dim=8, bias=False)

    if net_name == 'satellite_mlp':
        ae_net = MLP_Autoencoder(x_dim=36, h_dims=[32, 16], rep_dim=8, bias=False)

    if net_name == 'satimage-2_mlp':
        ae_net = MLP_Autoencoder(x_dim=36, h_dims=[32, 16], rep_dim=8, bias=False)

    if net_name == 'shuttle_mlp':
        ae_net = MLP_Autoencoder(x_dim=9, h_dims=[32, 16], rep_dim=8, bias=False)

    if net_name == 'thyroid_mlp':
        ae_net = MLP_Autoencoder(x_dim=6, h_dims=[32, 16], rep_dim=4, bias=False)   

    return ae_net
