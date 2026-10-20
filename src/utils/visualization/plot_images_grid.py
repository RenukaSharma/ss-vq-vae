import torch
import matplotlib
matplotlib.use('Agg')  # or 'PS', 'PDF', 'SVG'

import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid


def plot_images_grid(x: torch.tensor, export_img, title: str = '', nrow=8, padding=2, normalize=False, pad_value=0):
    """Plot 4D Tensor of images of shape (B x C x H x W) as a grid."""

    # print("x shape: ", x.shape)
    grid = make_grid(x, nrow=nrow, padding=padding, normalize=normalize, pad_value=pad_value)
    npgrid = grid.cpu().numpy()
    # print("grid: ", np.max(npgrid), np.min(npgrid), npgrid.shape)
    # t = np.transpose(npgrid, (1, 2, 0))
    # print("norm: ", np.linalg.norm(t[:, :, 0] - t[:, :, 2]))
    plt.imshow(np.transpose(npgrid, (1, 2, 0)))
    # plt.matshow(np.random.randn(50, 50, 3))
    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    if not (title == ''):
        plt.title(title)
    # plt.colorbar()
    plt.savefig(export_img, bbox_inches='tight', pad_inches=0.1)
    plt.clf()

def plot_matrix_grid(x: torch.tensor, export_img, title: str = '', nrow=8, padding=2, normalize=False, pad_value=0):
    """Plot 4D Tensor of images of shape (B x C x H x W) as a grid."""

    # print("x shape: ", x.shape)
    grid = make_grid(x, nrow=nrow, padding=padding, normalize=normalize, pad_value=pad_value)
    npgrid = grid.cpu().numpy()
    # print("grid: ", np.max(npgrid), np.min(npgrid), npgrid.shape)
    t = np.transpose(npgrid, (1, 2, 0))
    # print("norm: ", np.linalg.norm(t[:, :, 0] - t[:, :, 2]))
    plt.matshow(t[:, :, 0])
    # plt.matshow(np.random.randn(50, 50, 3))
    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    if not (title == ''):
        plt.title(title)
    plt.colorbar()
    plt.savefig(export_img, bbox_inches='tight', pad_inches=0.1)
    plt.clf()
