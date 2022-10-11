# Restor Foundation Tree Crown Delineation Pipeline

This repository contains a library for performing tree crown detection (TCD) in aerial imagery.

# Installation guidelines (for contributors)

Please set up your environment as follows:

1. Create a conda environment:

```bash
conda create -n tcd python=3.10
conda activate tcd
```

2. Install PyTorch from conda, this is necessary if you're using a new GPU like a GTX 3090 or an A100:

```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -c nvidia
```

If you need to check your CUDA version, run `nvidia-smi`:

```bash
> nvidia-smi

Tue Oct 11 18:17:57 2022       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 510.73.05    Driver Version: 510.73.05    CUDA Version: 11.6     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:0A:00.0  On |                  N/A |
|  0%   37C    P8    26W / 350W |   2343MiB / 24576MiB |      1%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
```

Here it's 11.6

3. Install the requirements from pip:

```bash
pip install -r requirements.txt
```

4. Setup git/git-lfs. This should be automatic if you already have LFS installed (e.g. on your first clone, git will download big files).

```bash
sudo apt install git-lfs
git lfs pull
```

Setup the pre-commit hooks:

```bash
pre-commit install
pre-commit autoupdate
```

This should run a few things before you push (import sorting, code formatting and notebook cleaning).