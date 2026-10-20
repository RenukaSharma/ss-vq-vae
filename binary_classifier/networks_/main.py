# from .mnist_LeNet import MNIST_LeNet, MNIST_LeNet_Autoencoder, MNIST_LeNet_Decoder
from .mnist_LeNet_vae import MNIST_LeNet, MNIST_LeNet_Autoencoder, MNIST_LeNet_Decoder

from .fmnist_LeNet import FashionMNIST_LeNet, FashionMNIST_LeNet_Autoencoder, FashionMNIST_LeNet_Decoder
# from .fmnist_LeNet_vae import FashionMNIST_LeNet, FashionMNIST_LeNet_Autoencoder, FashionMNIST_LeNet_Decoder

# from .cifar10_LeNet import CIFAR10_LeNet, CIFAR10_LeNet_Autoencoder, CIFAR10_LeNet_Decoder
from .cifar10_LeNet_vae import CIFAR10_LeNet
from .malaria_net import Malaria_net_Ext
from .binary_classifier_net import BinaryClassifierNet
from .nanofibre_LeNet_vae import Nanofibre_LeNet
# from .mlp import MLP, MLP_Autoencoder
# from .vae import VariationalAutoencoder
# from .dgm import DeepGenerativeModel, StackedDeepGenerativeModel



def build_autoencoder(net_name):
    """Builds the corresponding autoencoder network."""

    implemented_networks = ('mnist_LeNet', 'mnist_DGM_M1M2',
                            'fmnist_LeNet', 'fmnist_DGM_M1M2',
                            'cifar10_LeNet', 'cifar10_DGM_M1M2', 'malaria_net',
                            'arrhythmia_mlp', 'cardio_mlp', 'satellite_mlp', 'satimage-2_mlp', 'shuttle_mlp',
                            'thyroid_mlp', 'cifar10_classifier', 'mnist_classifier', 'fmnist_classifier', 'nanofibre_vae')

    assert net_name in implemented_networks

    ae_net = None

    if "classifier" in net_name:
        ae_net = BinaryClassifierNet(net_name)
    
    if net_name == 'mnist_LeNet':
        ae_net = MNIST_LeNet_Autoencoder()

    if net_name == 'mnist_DGM_M1M2':
        ae_net = VariationalAutoencoder([1*28*28, 32, [128, 64]])

    if net_name == 'fmnist_LeNet':
        ae_net = FashionMNIST_LeNet_Autoencoder()

    if net_name == 'fmnist_DGM_M1M2':
        ae_net = VariationalAutoencoder([1*28*28, 64, [256, 128]])

    if net_name == 'cifar10_LeNet':
        ae_net = CIFAR10_LeNet()
    
    if net_name == 'nanofibre_vae':
        ae_net = Nanofibre_LeNet()

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
