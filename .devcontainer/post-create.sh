#!/bin/bash
conda init
conda env create -f ./docker/environment.yaml
conda activate tcd
pip install torchmetrics==0.10.3
pip install -e .[test]
