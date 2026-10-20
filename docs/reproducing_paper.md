# Reproducing the DICTA 2023 (ss-VQ-VAE) experiments

This repository keeps the dataset loaders from the original research tree. Point `--data_path` (or `DATA_DIR`) at your local dataset root. The **active** malaria and corrosion loaders read tensors from `--data_path`.

## Environment

```bash
conda env create -f environment.yml
conda activate ss-vq-vae
```

Or from `src/`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r ../requirements.txt
```

Use a PyTorch build compatible with your CUDA driver if training on GPU. The pinned versions (PyTorch 1.1.0, Python 3.7) match the original research environment.

## Datasets

Point `--data_path` at a **local copy** of each dataset. Do not commit data into this repository.

### MVTec AD (`mvtec`, `net_name=vq_vae_net`)

Download from <https://www.mvtec.com/company/research/datasets/mvtec-ad/>. Pass `--category` (`carpet`, `bottle`, `leather`, `tile`, `wood`, object names, …). The paper reports bottle and carpet (and other MVTec categories in the thesis).

### Corrosion (`corrosion`)

The wired loader is `src/datasets/corrosion_patch_new.py`. Place these tensors **in** `--data_path`:

```
<data_path>/train_data_10k.pt
<data_path>/train_label_10k.pt
<data_path>/test_data_2p5k.pt
<data_path>/test_label_2p5k.pt
```

### Malaria (`malaria_dataset`)

The wired loader is `src/datasets/malaria_dataset_patch_with_GGnoise_noveltest.py`. Place these tensors **in** `--data_path`:

```
<data_path>/train_data_1209onwards_10k.pt
<data_path>/train_label_1209onwards_10k.pt
<data_path>/test_data_2p5k.pt
<data_path>/test_label_2p5k.pt
```

Older malaria files under `src/datasets/malaria_dataset_*.py` are not used by `load_dataset`.

### Nanofibre (`nanofibre`)

ImageFolder layout with `train/` and `test/` under `--data_path` (see `src/datasets/nanofibre.py`).

### MNIST / Fashion-MNIST / CIFAR-10

Downloaded by torchvision on first run if `--data_path` is writable.

## Training (ss-VQ-VAE)

The DICTA experiments use `net_name=vq_vae_net` and typically `--pretrain False`.

```bash
export DATA_DIR=$HOME/data/mvtec
export RESULTS_DIR=$HOME/results/ss-vq-vae
./scripts/run_experiment.sh mvtec vq_vae_net --category carpet --pretrain False
./scripts/run_experiment.sh --sweep corrosion vq_vae_net \
    --ratios 0,0.01,0.05,0.1,0.2 --seeds 0,1,2 --ablations A --pretrain False
```

Or from `src/` (Click requires `<xp_dir>` to exist):

```bash
python main.py mvtec vq_vae_net <xp_dir> <data_path> --category carpet --pretrain False [options...]
python main.py corrosion vq_vae_net <xp_dir> <data_path> --pretrain False [options...]
```

`--ablation_type {A,B,C,C_a,D,E,VAE,test,DeepSAD}` selects the loss configurations reported in the paper. `A` is the full proposed model.

Paper-style hyperparameters (from the original launchers): `--lr 1e-03`, `--n_epochs 2000`, `--batch_size 256` (textures) or `32` (objects), `--weight_decay 0.5e-6`, `--commitment_cost 0.25`, `--eta 1`.

## Hybrid SSAD baseline

```bash
python baseline_ssad.py malaria_dataset <xp_dir> <data_path> --hybrid True --load_ae /path/to/model.tar ...
```

`--load_ae` must match the architecture saved in `model.tar`.

## Binary classifier baseline

See `binary_classifier/` and `binary_classifier/baseline_binary_classifier.py`.

## t-SNE and visualization scripts

Plotters read `DATA_DIR` and `RESULTS_DIR` from the environment. Set those to your local folders before running `src/tsne_plotter_*.py` or `src/ss-VQ-VAE_visualize_*.py`.
