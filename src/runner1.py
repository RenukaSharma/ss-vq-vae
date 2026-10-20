from main import main
import click
import os



@click.command()
@click.argument('dataset_name', type=click.Choice(['mnist', 'fmnist', 'cifar10','malaria_dataset', 'arrhythmia', 'cardio', 'satellite',
                                                   'satimage-2', 'shuttle', 'thyroid']), required=False)
@click.argument('net_name', type=click.Choice(['mnist_LeNet', 'fmnist_LeNet', 'cifar10_LeNet', 'malaria_net','arrhythmia_mlp',
                                               'cardio_mlp', 'satellite_mlp', 'satimage-2_mlp', 'shuttle_mlp',
                                               'thyroid_mlp']), required=False)
@click.argument('xp_path', type=click.Path(exists=True), required=False)
@click.argument('data_path', type=click.Path(exists=True), required=False)
@click.option('--load_config', type=click.Path(exists=True), default=None,
              help='Config JSON-file path (default: None).')
@click.option('--load_model', type=click.Path(exists=True), default=None,
              help='Model file path (default: None).')
@click.option('--eta', type=float, default=1.0, help='Deep SAD hyperparameter eta (must be 0 < eta).')
@click.option('--ratio_known_normal', type=float, default=0.0,
              help='Ratio of known (labeled) normal training examples.')
@click.option('--ratio_known_outlier', type=float, default=0.0,
              help='Ratio of known (labeled) anomalous training examples.')
@click.option('--ratio_pollution', type=float, default=0.0,
              help='Pollution ratio of unlabeled training data with unknown (unlabeled) anomalies.')
@click.option('--device', type=str, default='cuda', help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')
@click.option('--seed', type=int, default=-1, help='Set seed. If -1, use randomization.')
@click.option('--optimizer_name', type=click.Choice(['adam']), default='adam',
              help='Name of the optimizer to use for Deep SAD network training.')
@click.option('--lr', type=float, default=0.001,
              help='Initial learning rate for Deep SAD network training. Default=0.001')
@click.option('--n_epochs', type=int, default=50, help='Number of epochs to train.')
@click.option('--lr_milestone', type=int, default=0, multiple=True,
              help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
@click.option('--batch_size', type=int, default=128, help='Batch size for mini-batch training.')
@click.option('--weight_decay', type=float, default=1e-6,
              help='Weight decay (L2 penalty) hyperparameter for Deep SAD objective.')
@click.option('--pretrain', type=bool, default=True,
              help='Pretrain neural network parameters via autoencoder.')
@click.option('--ae_optimizer_name', type=click.Choice(['adam']), default='adam',
              help='Name of the optimizer to use for autoencoder pretraining.')
@click.option('--ae_lr', type=float, default=0.001,
              help='Initial learning rate for autoencoder pretraining. Default=0.001')
@click.option('--ae_n_epochs', type=int, default=100, help='Number of epochs to train autoencoder.')
@click.option('--ae_lr_milestone', type=int, default=0, multiple=True,
              help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
@click.option('--ae_batch_size', type=int, default=128, help='Batch size for mini-batch autoencoder training.')
@click.option('--ae_weight_decay', type=float, default=1e-6,
              help='Weight decay (L2 penalty) hyperparameter for autoencoder objective.')
@click.option('--num_threads', type=int, default=0,
              help='Number of threads used for parallelizing CPU operations. 0 means that all resources are used.')
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
@click.option('--recon_param', type=float, default=0.5, help='The value for gamma(reconstruction parameter)')
@click.option('--eps', type=float, default=1e-4, help='eps')
@click.option('--tau', type=float, default=1e-4, help='tau')
@click.option('--delta', type=float, default=1e-4, help='delta')
@click.option('--const', type=float, default=1e-5, help='const')
@click.option('--ablation_type', type=click.Choice(['A', 'B', 'C', 'D', 'E', 'VAE', 'test']), default='A')

def mymain(dataset_name, net_name, xp_path, data_path, load_config, load_model, eta,
    ratio_known_normal, ratio_known_outlier, ratio_pollution, device, seed,
    optimizer_name, lr, n_epochs, lr_milestone, batch_size, weight_decay,
    pretrain, ae_optimizer_name, ae_lr, ae_n_epochs, ae_lr_milestone, ae_batch_size, ae_weight_decay,
    num_threads, n_jobs_dataloader, normal_class, known_outlier_class, n_known_outlier_classes, recon_param,
    eps, tau, delta, const, ablation_type):

    print(lr)
    ablation_type = 'B'
    dataset_name = 'cifar10'
    net_name = "cifar10_LeNet"
    xp_path = "../log_cifar10/run33"
    data_path = "../.."
    lr = 1e-4
    lr_milestone = [15, 30]
    n_epochs = 75
    pretrain = False
    normal_class = 1
    n_known_outlier_classes = 5
    # lens = [128, 512, 1000, 2000, 3000, 4000, 5000]
    # lens = [512]
    ratios = [0.2, 0.4, 0.6, 0.8, 1.0]
    seeds = [124, 273, 42, 300, 10]
    # seeds = [124]
    device = "cuda:3"
    length=5000
    for k in ratios:
        new_xp_path = xp_path + "/ratio_" + str(k)
        if not os.path.isdir(new_xp_path):
                os.makedirs(new_xp_path)
                print("Dir created")

        ratio_known_normal = (5*k)/6
        ratio_known_outlier = k/6
        ratio_pollution = 1/6 if k != 1.0 else 0

        for seed in seeds:
            temp_xp_path = new_xp_path + "/seed_" + ablation_type + "_" + str(seed)
            if not os.path.isdir(temp_xp_path):
                os.makedirs(temp_xp_path)
                print("Dir created")

            main(dataset_name, net_name, temp_xp_path, data_path, load_config, load_model, eta,
            ratio_known_normal, ratio_known_outlier, ratio_pollution, device, seed,
            optimizer_name, lr, n_epochs, lr_milestone, batch_size, weight_decay,
            pretrain, ae_optimizer_name, ae_lr, ae_n_epochs, ae_lr_milestone, ae_batch_size, ae_weight_decay,
            num_threads, n_jobs_dataloader, normal_class, known_outlier_class, n_known_outlier_classes, recon_param,
            eps, tau, delta, const, ablation_type, length)

# dataset_name = 'cifar10'
# net_name = "cifar10_LeNet"
# xp_path = "../log_cifar10/run26"
# data_path = "../.."

mymain()


