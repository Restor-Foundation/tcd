#!/bin/bash
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

conda init
conda env create -f ./docker/environment.yaml
conda activate tcd
pip install torchmetrics==0.10.3
pip install -e .[test]

# Download data + checkpoints and extract
gh release download init
mkdir -p checkpoints
mv *.ckpt checkpoints
mv *.pth checkpoints

# Extract dataset and rm archives
mkdir -p data
mv restor-tcd-oam* data
cd data && cat restor-tcd-oam-20221010.tar.gz.* | tar xzvf - && rm -rf *.tar.gz && cd ..

mv masks*.zip data/restor-tcd-oam
unzip data/restor-tcd-oam/masks*.zip -d data/restor-tcd-oam && rm -rf data/restor-tcd-oam/masks*.zip
