# Training models

## Quickstart

Models are trained using a variation of the following command:

```bash
tcd-train \
    model=semantic_segmentation/unet_resnet50 \
    data.root=/mnt/data/tcd/folds/holdout \
    data.output=/mnt/data/tcd/unet_r50/holdout
```

There are a couple of things that must be configured:

1. The type of model that is being trained, here UNet with a resnet50 backbone
2. The root folder that contains the dataset we're training on (with some standard/expected paths)
3. Where the results should be stored

Otherwise there are lots of things that can be customised by providing extra config key/value pairs, or by creating a new config file, but this is the basic command that we use for training all the models in the zoo.

For example, train a `segformer-mit-b5` model on a validation fold, with a learning rate of `1e-4`, batch size `1` and a tile size of `1024`:

```bash
tcd-train \
    model=semantic_segmentation/segformer \
    model.backbone=nvidia/mit-b5 \
    model.config.learning_rate=1e-4 \
    model.batch_size=1 \
    data.output=output/semantic/segformer-mit-b5/kfold4 \
    data.root=/mnt/data/tcd/kfold_4 \
    data.tile_size=1024
```

or a standard Mask-RCNN model on the holdout set:

```bash
tcd-train \
    model.config=detectron2/detectron_mask_rcnn.yaml \
    data.root=/mnt/data/tcd/holdout \
    data.output=/mnt/data/tcd/maskrcnn/holdout
```

## Dataset notes

Our dataset is provided in two formats: direct download as MS-COCO format from [Zenodo](LINK TBD) and as a HuggingFace [dataset](https://huggingface.co/datasets/restor/tcd). For training in the pipeline, we use MS-COCO format for compatibility (and easy integration with other libraries), but the HF dataset is provided for future work and we intend to integrate it soon.

We provide images as 2048x2048 GeoTIFF tiles. Each tile comes from an image downloaded from Open Aerial Map, resampled to 0.1 m/px. Images are annotated with instance labels in two classes: "tree" and "canopy" where "tree" represents an individual tree and canopy represents a group.

Since the dataset is not particularly large, although it is larger than many open tree detection datasets, we followed a k-fold cross-validation approach for model experimentation followed by testing on a holdout dataset. We stratify the dataset into 5 folds at the source image level (e.g. one "sample" in this stratification may map to multiple tiles) with an approximately equal distribution of biomes in each fold. Thus we can get a good idea of the consistency of the annotations. Our "release" models were trained on all data except the holdout split.

## Expected format

The training scripts expect you to pass a path to a folder with the following structure

```bash
/path/to/data
├── images
├── masks
├── test.json
├── train.json
└── val.json
```

where `images` contains images referenced in the annotation JSON files and `masks` contains identical filenames to `images`, but with `png` extensions.

## Generating MS-COCO datasets

We provide a convenience script in `tools/generate_dataset.py` to create the training dataset with the required folder structure. This script pulls the specified dataset from HuggingFace, typically `restor/tcd` and converts the Parquet records into an MS-COCO format dataset.

```bash
usage: generate_dataset.py [-h] [--tiffs] [--folds] dataset [dataset ...] output

Convert a Restor TCD dataset on HuggingFace to MS-COCO format.

positional arguments:
  dataset     List of dataset names, to be downloaded from HF Hub
  output      Output path where the dataset will be created

options:
  -h, --help  show this help message and exit
  --tiffs     Output images as GeoTIFF instead of JPEG (may slow down dataloading)
  --folds     Generate cross-validation folds based on indices in the dataset
```

By default the converter will generate JPEG images as these can be much faster to load during training than tiffs. However all the information is present in the HuggingFace dataset to create GeoTIFF files, which you can create if you use the `--tiffs` option. This may be helpful if you want to do some geospatial analysis on the dataset.

If you pass the `--folds` option, the script will generate cross-validation folds for the dataset. We use this approach to better estimate the performance of the dataset given the limited number of images. Note that if you do decide to choose your own validation approach, you **must** ensure that you split based on the `oam_id` field to make sure that you don't have tiles from the same image in different folds (this will skew your results). For example, suppose you wanted to use a standard 80/20 train/val split:

```python
import datasets
from sklearn.model_selection import train_test_split

dataset = datasets.load_dataset('restor/tcd')

# Get unique IDs in the dataset
ids = list(set(dataset['train']['oam_id']))

# Split the IDs
train_ids, val_ids = train_test_split(ids, test_size=0.2, random_state=42)

# Generate the train and validation subsets
train_ds = dataset.filter(lambda x: x['oam_id'] in train_ids)
val_ds =  dataset.filter(lambda x: x['oam_id'] in val_ids)

# Save datasets + train ...
```

This is essentially the approach we used to split the dataset into folds, except we use the biome indices for stratification, for example:

```python
from sklearn.model_selection import StratifiedKFold

# Make a mapping between OAM IDs and biomes
id2biome = {}
for row in dataset['train'].select_columns(['oam_id', 'biome']):
    id2biome[row['oam_id']] = row['biome']

biomes = [id2biome[i] for i in ids]

splitter = StratifiedKFold(n_splits=2)
train_folds, val_folds = splitter.split(ids, biomes) # Returns a list of ID lists, one for each fold
```

When you train models on the data, just pass the option `data.root=/path/to/output`. Note that the training scripts consider `train.json` and `val.json` and ignore `test.json`. The idea in cross-validation with a holdout set is that we _don't look_ at the holdout data until we've decided that we've finished tuning on the validation folds.

For the holdout set, we periodically evaluate on the `test.json` file during training so we can take the best checkpoint (since we still need some independent measure of performance as we train). This sounds like bad practice, but this assumes we've already completed our cross-validation. In principle we only train on the full dataset a single time per model.

## Training parameters

Training is performed with a fixed random seed (`42`) which should lead to more-or-less deterministic results. There is some randomness which is quite hard to get rid of, but you should find that if you run the training scripts on your own machine you get similar performance. The following hyperparameters were used:

- Seed: `42` for repeatability
- Folds/split: as described above, we provide train/val JSON files for each split as well as a test (holdout) JSON for final evaluation
- Learning rate: adjusted via basic grid search for performance and then fixed (this is slightly different for each model)
- Learning rate schedule: simple stepped approach with "decrease on plateau"; this is an approach that the Detectron team has used in almost all their models and it seems to work fine.
- Pre-training: we found that using models pre-trained on MS-COCO converged faster, though the training curves suggest that from-scratch would also work if left long enough.
- Image size: we train on 1024 px (i.e. a random fixed-size crop from an augmented image). Model resize stages are disabled to ensure we train on the desired spatial scale.
- Batch size: adjusted for available GPU RAM. For semantic segmentation models we use gradient accumulation for an effective batch size of 32
- Epochs: when performing cross-validation we trained for 75 epochs to ensure convergence. We noted that convergence was achieved within 50 epochs, so we trained the release models with this reduced schedule unless the training curves suggested we should push for longer.
- Augmentation: we augment images in a variety of ways (flipping, rotation, brightness adjustments, etc)
- Quantisation: we trained all our models in FP32
- Loss: for Detectron2 we use the built-in Mask-RCNN loss. For segmentation with SMP models we use focal loss, for SegFormer we use the built-in loss; in both cases we track a number of statistics during training including IoU/Jaccard Index, Accuracy and Precision, Recall and F1.

For exact specifications you can check the configuration files provided with each checkpoint. Due to library improvements some parameters may have been moved or renamed since training, but we try to provide up-to-date training configs for all model variants.

We did not spend a huge amount of time performing hyper-parameter tuning beyond simple things like learning rate and we encourage the community to experiment. Given the sheer variety of models available, we chose architectures that we found to be simple to train and provided good performance. SegFormer, in particular, proved to be an excellent segmentation architecture with relatively few parameters compared to older baselines like UNet. We found that training transformer-based instance segmentation models was more challenging and the training to be more brittle than with CNN-based architectures. Mask-RCNN appears to perform well within the limitations of the dataset and its annotations, but there is probably some value in exploring other architectures like PointRend or Mask2Former.

## Instance Segmentation

Training code for instance segmentation is provided for Mask-RCNN. While there are numerous instance detection architectures, Mask-RCNN is well-studied and Detectron2 has proved to be a reliable and robust training framework without needing lots of hyperparameter turning. Newer transformer-based architectures show promise, but we have struggled to get models to train reliably and efficiently. The dataset on HuggingFace is broadly compatible with the models in the `transformers` library, particularly with regard to mask formats.

## Semantic Segmentation

We provide training code for UNet and Segformer models. UNet is a classic architecture that is well understood, has no licesning issues, and - in our case - performs well. Segformer is a newer transformer-based architecture. It has a stricter license than UNet as provided by Nvidia, though users can apply (to NVIDIA) to use it for commercial purposes. Thus we recommend that you only use the segformer models for personal and research use.

## Resuming training

!!! warning

    Model training resumption is supported for instance segmentation models only, currently. There is a known bug with state-loading with Pytorch Lightning which causes losses to spike after resuming training - we're looking into this and will push a fix when we determine what the problem is. For now, if a segmentation training jobs fails it needs to be started from scratch.

To resume training, pass `model.resume=1` to the training script and set the output directory to the folder that stored the last checkpoint. Note that if you want to override something like the number of iterations (e.g. to continue training), you need to edit the config file that is stored in the output folder (all the hyperparamters will be reloaded from that file with the assumption that you just want to resume a run that crashed for some reason). Double check in the logs that the model training restarts from the expected iteration.

For example:

```bash
tcd-train model=instance_segmentation/default model.config=detectron2/detectron_mask_rcnn.yaml model.resume=1 data.root=/home/josh/data/tcd/kfold_4 data.output=/mnt/internal/data/tcd/maskrcnn_r50/kfold_4/20240101_0101/
```

### Typical training curves

You can see training metrics on the HuggingFace pages for each model.

In most cases the training loss is correlated with the parameter count of the model (e.g. b5 is better than b0). For validation, the results are a bit less clear and suggest that we are running into the limits of the annotations in the dataset. Larger models perform better, but not by a huge amount; for larger jobs where throughput/latency is important you may be fine running the smallest models. This is also the case in environments where we expect the model to perform very well, like urban canopy coverage.

## Evaluation

Model evaluation is straightforward if you use the `pipeline` method, for example here we take a "best" model from a training run and evaluate it on one of the validation data folds:

```python
pipeline = Pipeline("instance",
                  overrides="model.weights=/home/josh/data/data/tcd/maskrcnn_r50/kfold_4/20240402_0806/model_best.pth")

pipeline = runner.evaluate(
    annotation_file="/home/josh/data/tcd/kfold_4/val.json",
    image_folder="/home/josh/data/tcd/kfold_4/images",
    output_folder="/home/josh/data/.cache/kfold4",
)
```

In general it's important that you run evaluation separately after training, especially if reporting results for publication. By default, Detectron2 will evaluate on the _final_ model checkpoint, not the _best_. The approach above will make sure that you run on the weights that you intend to. If you have predictions already, then you can provide `prediction_file` which should be a path to a MS-COCO formatted JSON file. This will run COCO evaluation on its own, using Detectron's modified evaluator that supports an arbitrary number of detections per image.

For semantic segmentation, the pipeline uses the Pytorch Lightning datamodule system and whatever you've specified as the "test" split (i.e. the `test.json` file in your data directory), minimally the following should work - you can customise your pipeline as you would for normal predictions (e.g. specify overrides).

```python
pipeline = Pipeline('semantic', overrides=['data.root=/home/josh/data/tcd/holdout'])
pipeline.evaluate()
```

### Independent evaluation

We provide an evaluation script (`evaluate.py`) that you can use for evaluating against your own data for the following scenarios:

- Segmentation vs Ground truth: typically used for comparing model outputs to Canopy Height Models. The script handles differences in source resolution.
- Instance vs instance: simplest method is to convert your ground truth to MS-COCO format and then evaluate as above. This will work even if you have a single image.
- Instance vs keypoint: performs a point-in-polygon test for each keypoint and reports statistics on the results. This sort of evaluation can be used if you have tree crown centres, but not masks. Often the keypoints will be a subset of all the trees in the image, so some metrics like false positives can be ignored if this is the case. This is the approach we use for evaluating the Zurich and WeRobotics/Tonga datasets. The inputs are typically a shapefile with instance polygons and a list of (x,y) points.

We plan to provide the following options in the future:

- Instance vs bounding box: you can also do this with the `evaluate` method, but you need to change the task type to `box` instead of `segm`. This will treat model predictions as boxes and ignore the masks.
- Semantic vs keypoint: a somewhat tenuous evaluation metric, but similar to the instance approach - checks whether classified keypoints align with the predicted semantic mask. You can optionally set a radius around each keypoint that will be used to confirm.
- Semantic vs geometry (polygon): similar to the above, computes statistics for each geometry to determine whether the prediction is correct. Bounding boxes are not great here, because they will typically contain background pixels.

## Exporting models for production use

TBD

## Benchmarks

Our dataset is relatively unique in the literature in that we provide instance-level masks. Most other datasets either provide tree locations (i.e. single points) or bounding boxes (DeepForest). This makes an independent apples-to-apples evaluation difficult. Cross-validation statistics on our own dataset (see the paper) show that our models perform well and that our labels are probably self-consistent, and we have verified this using partner data.

We provide a few independent benchmarks:

- Detecting tree instances and canopy cover in the city of Zurich using public 10 cm SwissTopo data. The city provides both a high resolution CHM and tree locations for 20k municipal trees.
- Predictions on a large orthomosaic from Tonga with numerous individual trees annotated.

However we believe that the models speak for themselves - you are encouraged to provide your own imagery to try out. We welcome criticism and feedback if you find that the models aren't predicting as well as you expect. If you have labelled imagery, we also encourage you to share it so that we can improve the model.
