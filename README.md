# Restor Foundation Tree Crown Delineation Pipeline

![Coverage Status](coverage-badge.svg)

This repository contains a library for performing tree crown detection (TCD) in aerial imagery.

# Dataset/model setup

You can find dataset and models on the [release](https://github.com/Restor-Foundation/tcd-pipeline/releases/latest) page.

Download a model file an extract it to `<repo dir>/checkpoints/model_final.pth`, this is the default path but you can modify the configuration file if you save it elsewhere.

Similarly, download the dataset which contains images and COCO format annotations. You can join the archives together and extract with the following command:

```
cat restor-tcd-oam-20221010.tar.gz.* | tar xzvf -
```

By default, you should place in this in the `<repo-dir>/data` folder. If you put it somewhere else, update the configuration file if you want to run a new training or evaluation job.

# Installation guidelines (for contributors)

Please set up your environment as follows:

1. Create a conda environment:

```bash
conda create -n tcd python=3.10
conda activate tcd
```

If you're running an M1 Mac:

```
CONDA_SUBDIR=osx-arm64 conda create -n tcd python=3.10
conda activate tcd
conda config --env --set subdir osx-arm64
```

2. Install PyTorch from conda, this is necessary if you're using a new GPU like a GTX 3090 or an A100:

```bash
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch -c nvidia -y
```

If you're running on a Mac or a CPU-only machine, omit `cudatoolkit`. If you are running an M1 Mac you can use pip:

```bash
pip install torchvision torch
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

Here it's 11.6, but this may not yet be compatible with pytorch (unless you build from source), so stick to 11.3 for now. When you run install, double check the output looks something like this:

```
pytorch            pytorch/linux-64::pytorch-1.12.1-py3.10_cuda11.3_cudnn8.3.2_0 None
pytorch-mutex      pytorch/noarch::pytorch-mutex-1.0-cuda None
torchaudio         pytorch/linux-64::torchaudio-0.12.1-py310_cu113 None
torchvision        pytorch/linux-64::torchvision-0.13.1-py310_cu113 None
```

(says `cuda/cuxxx` instead of `cpu`). The single image demo in the notebook should run fairly quickly on a modern GPU (e.g. around 1 second per iteration on a 3090/A100. You can do even faster if test-time augmentation is disabled).

3. Install GDAL:

```bash
conda install rasterio fiona gdal -c conda-forge -y
```

4. Install the remaining requirements from pip:

```bash
pip install -r requirements.txt
```

Setup the pre-commit hooks:

```bash
pre-commit install
pre-commit autoupdate
```

This should run a few things before you push (import sorting, code formatting and notebook cleaning).

## Testing

You can run the test suite here:

```
python -m pytest
coverage xml -o ./reports/coverage/coverage.xml

```

This can be a bit slow to run on a CPU with TTA and tiling checks. Also run:

```
coverage-badge -f -o coverage-badge.svg
```

to generate badges.