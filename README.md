# Restor Foundation Tree Crown Delineation Pipeline

![Coverage Status](coverage-badge.svg)
![CI Status](https://github.com/jveitchmichaelis/tcd/actions/workflows/python-test.yml/badge.svg)
![Docker Build](https://github.com/jveitchmichaelis/tcd/actions/workflows/docker.yml/badge.svg)

This repository contains a library for performing tree crown detection (TCD) in aerial imagery.

# Dataset/model setup

You can find dataset and models on the [release](https://github.com/Restor-Foundation/tcd-pipeline/releases/latest) page.

Download a model file an extract it to `<repo dir>/checkpoints/model_final.pth`, this is the default path but you can modify the configuration file if you save it elsewhere.

Similarly, download the dataset which contains images and COCO format annotations. You can join the archives together and extract with the following command:

```
cat restor-tcd-oam-20221010.tar.gz.* | tar xzvf -
```

By default, you should place in this in the `<repo-dir>/data` folder. If you put it somewhere else, update the configuration file if you want to run a new training or evaluation job.

# General installation guidelines

Please set up your environment as follows:

1. Create a conda environment:

```bash
conda create -n tcd python=3.10
conda activate tcd
```

If you're running a Mac with Apple Silicon (M1/2/3) this is very important, otherwise performance will tank:

```
CONDA_SUBDIR=osx-arm64 conda create -n tcd python=3.11
conda activate tcd
conda config --env --set subdir osx-arm64
```

2. Install PyTorch and the cuda toolkit. It's important to do this as a single step, because the toolkit (`cuda`) isn't included with pytorch and if you do it separately, you may get version mismatches. Also we need to update the compiler. This should give you a totally isolated environment to work with:

```bash
conda install cuda pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
conda install gcc=12.3.0 gxx=12.3.0 libstdcxx-ng=12.3.0 -c conda-forge
```

If you're running on a Mac or a CPU-only machine, omit `cudatoolkit`. If you are running on a Mac you can use pip:

```bash
python -m pip install torchvision torch
```

3. Install GDAL:

```bash
conda install rasterio fiona gdal -c conda-forge -y
```

There may be issues with certain library dependencies on GDAL, so installing from Conda may be preferable.

4. Install the remaining requirements from pip. You should make sure you have torchmetrics==0.10.3 as there are some compatibility issues and breaking changes in 0.11 (plus bugs in the Dice loss).

```bash
python -m pip install -r requirements.txt
```

Setup the pre-commit hooks:

```bash
pre-commit install
pre-commit autoupdate
```

This should run a few things before you push (import sorting, code formatting and notebook cleaning).

5. Install locally in editable mode (useful for testing)

```
python -m pip install -e .
```

## Post-install and testing

The simplest way to check that everything is working is to run the test suite here:

```
python -m pytest
coverage xml -o ./reports/coverage/coverage.xml

```

This can be a bit slow to run on a CPU with TTA and tiling checks. Also run:

```
coverage-badge -f -o coverage-badge.svg
```

to generate badges.

## Training models

You can use the `launch.py` command to run various jobs like training or prediction. For example, to train a semantic segmentation model:

```python
python launch.py job=train model=semantic_segmentation/unetplusplus_resnet50 data.output=/media/josh/data/tcd/unet_r50/kfold4 data.root=/home/josh/data/tcd/kfold_4 data.tile_size=1024
```

We need to specify the `task` (train), the model type (`unet_resnet50`), the output folder (`data.output`) and here we also override a couple of settings like the tile size (`1024`).


## Train a semantic segmentation model

Assuming the dataset is extracted to `data/restor-tcd-oam`. First, from the root directory, generate masks (if you run from elsewhere, just update the paths):

```bash
python tools/masking.py --images data/restor-tcd-oam/images --annotations data/restor-tcd-oam/train_20221010.json --prefix train
python tools/masking.py --images data/restor-tcd-oam/images --annotations data/restor-tcd-oam/val_20221010.json --prefix val
python tools/masking.py --images data/restor-tcd-oam/images --annotations data/restor-tcd-oam/test_20221010.json --prefix test
```

This will generate binary segmentation masks for every image in the dataset.

To train a model, then simply run:

```
from tcd_pipeline.pipeline import Pipeline

runner = Pipeline("config/train_kfold0_semantic.yaml")
runner.train()

```

You can replace `kfold0` with [0-5]. Since data labelling is expensive, we prefer to use k-fold cross validation rather than a standard 80/10/10 train/val/test split.

## Train an instance segmentation model

Follow the guidance above, but replace the config with `config/train_kfold0_detection.yaml`.

## Evaluation

Once you've trained or loaded a model, you can call the `.evaluate()` method on a `ModelRunner` object to perform an evaluation run on the test dataset.

## Fixup for old releases

Due to some API changes in other libraries, some older checkpoints may now break. Since we want to take advantage of newer versions of Lightning and Torch, you can upgrade these checkpoints to have them work. Run:

```bash
python tools/fixup_checkpoint.py ./checkpoints/unet_resnet34.ckpt ./checkpoints/unet_resnet34_2.ckpt
mv ./checkpoints/unet_resnet34_2.ckpt ./checkpoints/unet_resnet34.ckpt
```

## Generating a TCD report

The simplest "entrypoint" to the pipeline is to use the `predict_report.py` tool in `tools`. 

```bash
> python predict_report -h

usage: predict_report.py [-h] -i IMAGE [-g GEOMETRY] [-s TILE_SIZE] [-o OUTPUT] [-r] [--semantic_only] [--overwrite] [--gsd GSD] [--instance_seg INSTANCE_SEG] [--semantic_seg SEMANTIC_SEG]

options:
  -h, --help            show this help message and exit
  -i IMAGE, --image IMAGE
                        Input image (GeoTIFF orthomosasic)
  -g GEOMETRY, --geometry GEOMETRY
                        Input shapefile (GeoJSON or Shapefile)
  -s TILE_SIZE, --tile-size TILE_SIZE
                        Tile size
  -o OUTPUT, --output OUTPUT
                        Working and output directory
  -r, --resample        Resample image
  --semantic_only       Only perform semantic segmentation
  --overwrite           Overwrite existing results, otherwise use old ones.
  --gsd GSD
  --instance_seg INSTANCE_SEG
                        Instance segmentation config
  --semantic_seg SEMANTIC_SEG
                        Semantic segmentation config
```

Generally you can run:

```bash
python tools/predict_report.py -i data/5c15321f63d9810007f8b06f_10_00000.tif
```

and you should see something like:

```bash
(tcd) tcd-pipeline % python tools/predict_report.py -i data/5c15321f63d9810007f8b06f_10_00000.tif --overwrite
INFO:__main__:Storing output in data/5c15321f63d9810007f8b06f_10_00000_pred
WARNING:__name__:Failed to use CUDA, falling back to CPU
INFO:__name__:Device: cpu
  0%|                                                                                                                                                                                           | 0/1 [00:00<?, ?it/s]INFO:root:Loading checkpoint: /Users/josh/code/tcd-pipeline/checkpoints/unet_resnet34.ckpt
INFO: Lightning automatically upgraded your loaded checkpoint from v1.8.3.post0 to v2.1.0. To apply the upgrade to your files permanently, run `python -m lightning.pytorch.utilities.upgrade_checkpoint checkpoints/unet_resnet34.ckpt`
INFO:lightning.pytorch.utilities.migration.utils:Lightning automatically upgraded your loaded checkpoint from v1.8.3.post0 to v2.1.0. To apply the upgrade to your files permanently, run `python -m lightning.pytorch.utilities.upgrade_checkpoint checkpoints/unet_resnet34.ckpt`
() {'in_channels': 3, 'num_classes': 2, 'loss': 'focal', 'ignore_index': None, 'model': 'unet++', 'backbone': 'resnet34', 'weights': 'imagenet', 'lr': 0.001, 'patience': 5}
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:19<00:00, 19.03s/it, #objs: 2, CPU: 1.81G, t_pred: 18.94s, t_post: 0.01s]
INFO:__name__:Processing cached results
INFO:tcd_pipeline.post_processing:Looking for cached files in: /Users/josh/code/tcd-pipeline/temp/5c15321f63d9810007f8b06f_10_00000_cache
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 59.35it/s]
INFO:tcd_pipeline.post_processing:Result collection complete
INFO:__name__:Cleaning up post processor
INFO:tcd_pipeline.result:Serialising results to data/5c15321f63d9810007f8b06f_10_00000_pred/semantic_segmentation/results
WARNING:__name__:Failed to use CUDA, falling back to CPU
INFO:__name__:Device: cpu
  0%|                                                                                                                                                                                           | 0/1 [00:00<?, ?it/s]INFO:detectron2.checkpoint.detection_checkpoint:[DetectionCheckpointer] Loading from /Users/josh/code/tcd-pipeline/checkpoints/model_final.pth ...
INFO:fvcore.common.checkpoint:[Checkpointer] Loading from /Users/josh/code/tcd-pipeline/checkpoints/model_final.pth ...
INFO:__name__:Using Test-Time Augmentation
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:28<00:00, 28.65s/it, #objs: 418, CPU: 6.51G, t_pred: 28.48s, t_post: 0.10s]
INFO:__name__:Processing cached results
INFO:tcd_pipeline.post_processing:Looking for cached files in: /Users/josh/code/tcd-pipeline/temp/5c15321f63d9810007f8b06f_10_00000_cache
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 163.46it/s]
INFO:tcd_pipeline.post_processing:Running non-max suppression
INFO:tcd_pipeline.post_processing:Result collection complete
INFO:__name__:Cleaning up post processor
INFO:tcd_pipeline.result:Serialising results to data/5c15321f63d9810007f8b06f_10_00000_pred/instance_segmentation/results.json
loading annotations into memory...
Done (t=0.00s)
creating index...
index created!
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 409/409 [00:00<00:00, 4760.45it/s]
INFO:root:Generating report
Loading pages (1/6)
Counting pages (2/6)                                               
Resolving links (4/6)                                                       
Loading headers and footers (5/6)                                           
Printing pages (6/6)
Done                                      
``````