# Installation

We provide a few methods to install the pipeline:

- `conda` is the preferred method of installing the pipeline in an isolated environment with all the dependencies that you need.
- `pip` can also be used with a virtual environment manager like `virtualenv` or `pyenv`
- `docker` container - the most robust as it's completely self-contained

### Getting the repository

You can clone the repository using `git`, we recommend cloning via SSH:

```bash
git clone git@github.com:restor-foundation/tcd
```

### CUDA

As with most deep learning models, our tree crown detection pipeline will run much faster if you have some kind of ML accelerator installed on your system. Typically this is an NVIDIA GPU with CUDA drivers installed. If you don't have a GPU, you can still run the pipeline on a CPU but it will be slower.

For the installation process below, we typically install a specific version of CUDA inside a virtual environment to avoid conflicts with other things on your system. Thus while you _probably_ have CUDA installed if you own a capable GPU, the methods below will install standalone CUDA runtimes which should not interfere with whatever's on your system.

### Dependency overview

- An up-to-date PyTorch install. We officially support version 2 or later, but version 1 will probably work.
- [GDAL](https://gdal.org/) for reading and writing geospatial data
- Dependencies as listed in the `requirements.txt` file
- For model support:
    - [Detectron2](https://detectron2.readthedocs.io/), which is required to run instance segmentation models
    - The `transformers` and `datasets` [libraries](https://huggingface.co), which are used for model + data hosting and semantic segmentation models
    - [Segmentation Models Pytorch](https://segmentation-modelspytorch.readthedocs.io/en/latest/) for UNet and other CNN-based segmentation architectures
- For training, [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/)

Of these, the most challenging to get "right" is a working Detectron2 install.

If you're using a GPU, you should make sure that PyTorch is installed with CUDA support. If you're using a CPU, you can install PyTorch without CUDA support. If you're using an ARM Mac then you can also use the `mps` backend.

## Conda

Conda is an environment and package manager that is widely used within the Python community. It can create a fully isolated environment that includes system packages, and not just Python libraries.  You can install `conda` using the instructions [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

We provide a frozen `conda` environment file that you can install (on Linux) as follows:

```bash
conda env create --name tcd --file environment.yml
```

and then install the pipeline:

```bash
pip install -e .[test]
```

If you run into issues when installing Detectron2 that `nvcc` isn't detected (typically related to the wrong version of `CUDA` being located), then try the following:

```bash
conda activate tcd
export CUDA_HOME=$CONDA_PREFIX
conda env update --name tcd --file environment.yml
```

### Building your own conda env

If you need to create your own environment, or you want to try different versions of libraries then this is the process we use to create the `environment.yml` file. First, create and activate an environment:

```bash
conda create -n tcd python=3.12
conda activate tcd
```

!!! apple "Using conda on OSX"

    On Macs with ARM processors (e.g. M1/M2/M3), you should make sure that `conda` uses packages appropriate for your architecture:

    ```bash
    CONDA_SUBDIR=osx-arm64 conda create -n tcd python=3.12
    conda activate tcd
    conda config --env --set subdir osx-arm64
    ```

For Linux/Windows users, install pytorch and CUDA:

```bash
conda install pytorch torchvision pytorch-cuda=12.1 cuda=12.1 -c pytorch -c nvidia
conda install gcc=12.3.0 gxx=12.3.0 libstdcxx-ng=12.3.0 -c conda-forge
```

At this point, you should check that the CUDA home path is inside the conda environment:

```bash
which nvcc # should return something like /home/josh/miniconda3/envs/tcd/bin/nvcc
```

!!! apple "Installing torch on OSX"

    On Macs you don't need to worry about CUDA:

    ```bash
    conda install pytorch torchvision -c pytorch
    ```

!!! question "Wait, isn't pytorch-cuda enough?"

    The reason that `cuda` is included here is that it's a requirement to build Detectron2, which depends on `nvcc`, the CUDA compiler. It's important that you install the same version of CUDA that PyTorch is expecting, so here we install everything in one step so that `conda` solves the dependencies for us. You may not need to specify the version of the `cuda` package, as long as you install it at the same time as `pytorch`.

    We are working on a release of the pipeline that doesn't require Detectron2 to be installed, as it's a relatively heavy dependency.

Assuming you've cloned the repository, you can then install the requirements using `pip` (in principle you can also do this directly with `conda`, but `pip` is generally a lot faster).

```bash
pip install -r requirements.txt
```

Now you can export the environment to a file:

```bash
conda env export -f environment.yml
```

and install the pipeline itself:

```bash
pip install -e .[test]
```

!!! note
    You will need to adjust the `detectron2` requirement in the environment as it won't be a git path any more, and if you freeze the environment after installing the pipeline, you will also need to remove that because when you re-install from the environment, pip won't know where to locate it. This will be fixed once we release a pip-installable package for the pipeline.

## Using pip, pyenv, etc.

!!! warning

    It is strongly recommended that you use pip inside a virtual or conda environment. If you're on Linux, **do not install the pipeline into your system environment** (recent versions of Debian/Ubuntu will warn against this). That said, there are some times when this may be appropriate. 

`conda` isn't always appropriate and it can be difficult to work with inside containers or other lean environments. Note for this method to work, you do need a system-wide CUDA install such that `nvcc` can be found. This is the approach that we use [on GitHub for automated testing](https://github.com/restor-foundation/tcd/actions/workflows/python-test.yml) and within Docker.

Here's how to install the pipeline using "plain" Python with `virtualenv` (we also install GDAL which is required for some libraries like `rasterio`):

```bash
sudo apt install python3 python3-virtualenv gdal-bin
python3 -m venv tcd_env
source tcd_env/bin/activate
pip install --upgrade pip setuptools wheel
```

!!! apple "Getting GDAL on Mac"

    If you run into dependency problems on Mac, [Homebrew](https://brew.sh) is normally your friend:

    ```bash
    brew install python virtualenv gdal
    ```


!!! warning

    Don't be tempted to omit the last `pip install` command. Having `wheel` is critical to install Detectron2 and it turns out that it's not always included in base Python environments. If you see the following error when trying to install the requirements, it's probably because you didn't install `wheel` first.

    ```bash
    ModuleNotFoundError: No module named 'torch'
    ```


Then, install `torch` using the [latest instructions](https://pytorch.org/get-started/locally/) from Pytorch's website. Torch should detect your system CUDA installation:

```bash
pip install torch torchvision
```

check that `torch` is installed with CUDA support, if you were expecting it:

```bash
$ python
> import torch
> torch.cuda.is_available()
True
```

then as above, install the pipeline and requirements:

```bash
pip install -r requirements.txt
pip install -e .[test]
```

## Docker

We provide Dockerfiles that have the pipeline pre-installed. We use the `pytorch/pytorch` base image which comes with CUDA support and has `torch` already installed. The Dockerfile simply adds the library dependencies on top and contains a clone of the repository.

### Pulling from Github Container Repo

TBD.

### Building containers:

```bash
cd docker
./build.sh Dockerfile
```

on ARM64 (e.g. Mac M series with ARM silicon):

```bash
cd docker
./build.sh Dockerfile_arm
```

## Verifying the install

The most comprehensive way to check that you've installed everything is to run the test suite from the root directory of the repository:

```bash
pytest
```

In the process of running the tests, the training dataset will be downloaded and cached, as well as most of the models. This will take between 5-10 GB of disk space. When we release updates it's important that we check that the dataset can be automatically obtained, but if you don't want/need it, you can skip to the [next section](prediction.md) and try to run some predictions instead.


## Building docs

The documentation pages you're reading now are also included in the repository, in order to build them you can run:

```bash
pip install -e .[docs]
```

and then

```bash
mkdocs serve
```

We use [MkDocs](https://www.mkdocs.org/) with the beautiful [Material](https://squidfunk.github.io/mkdocs-material/) theme which provides a lot of very nice features for markup.

## What next?

Once you've installed the pipeline, it's time to try out some models! Head on over to the [prediction](prediction.md) documentation.