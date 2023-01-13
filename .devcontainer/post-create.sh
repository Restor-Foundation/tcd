#!/bin/bash
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

mkdir -p data
mv restor-tcd-oam* data
cd data && cat restor-tcd-oam-20221010.tar.gz.* | tar xzvf - && cd ..
