# Training models

## Quickstart

Models are trained using a variation the following command:

```bash
python train.py \
    model.config=semantic_segmentation/unet_resnet50 \
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
python train.py \
    model=semantic_segmentation/segformer \
    model.config.backbone=nvidia/mit-b5 \
    model.config.learning_rate=1e-4 \
    model.batch_size=1 \
    data.output=output/semantic/segformer-mit-b5/kfold4 \
    data.root=/mnt/data/tcd/kfold_4 \
    data.tile_size=1024
```

or a standard Mask-RCNN model on the holdout set:

```bash
python train.py \
    model.config=detectron2/detectron_mask_rcnn.yaml \
    data.root=/mnt/data/tcd/holdout \
    data.output=/mnt/data/tcd/maskrcnn/holdout
```

## Dataset notes

Our dataset is provided in two formats: direct download as MS-COCO format from [Zenodo](LINK TBD) and as a HuggingFace [dataset](https://huggingface.co/datasets/restor/tcd). For training in the pipeline, we use MS-COCO format for compatibility (and easy integration with other libraries), but the HF dataset is provided for future compatibility.

We provide images as 2048x2048 GeoTIFF tiles. Each tile comes from an image downloaded from Open Aerial Map, resampled to 0.1 m/px. Images are annotated with instance labels in two classes: "tree" and "canopy" where "tree" represents an individual tree and canopy represents a group.

Since the dataset is not particularly large, although it is larger than many open tree detection datasets, we followed a k-fold cross-validation approach for model experimentation followed by testing on a holdout dataset. We stratify the dataset into 5 folds at the source image level (e.g. one "sample" in this stratification may map to multiple tiles) with an approximately equal distribution of biomes in each fold. Thus we can get a good idea of the consistency of the annotations. Our "release" models were trained on all data except the holdout split.

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

We did not spend a huge amount of time performing hyper-parameter tuning beyond simple things like learning rate and we encourage the community to experiment. Given the sheer variety of models available, we chose architectures that we found to be simple to train and provided good performance. SegFormer, in particular, proved to be an excellent segmentation architecture with relatively few parameters compared to older baselines like UNet. We found that training transformer-based instance segmentation models was more challenging and the training to be more brittle than with CNN-based architectures. Mask-RCNN appears to perform well within the limitations of the dataset and its annotations, but there is probably some value in exploring other architectures like PointRend.

## Instance Segmentation

Training code for instance segmentation is provided for Mask-RCNN. While there are numerous instance detection architectures, Mask-RCNN is well-studied and Detectron2 has proved to be a reliable and robust training framework without needing lots of hyperparameter turning. Newer transformer-based architectures show promise, but we have struggled to get models to train reliably and efficiently. The dataset on HuggingFace is broadly compatible with the models in the `transformers` library, particularly with regard to mask formats.

## Semantic Segmentation

We provide training code for UNet and Segformer models. UNet is a classic architecture that is well understood and - in our case - performs well. Segformer is a newer transformer-based architecture. It has a stricter license than UNet as provided by Nvidia, though users can apply for a license to use it for commercial purposes. Thus we recommend that by default you only use the segformer models for personal and research use.

## Resuming training

Model training resumption is supported for instance segmentation models only, currently. There is a known bug with state-loading with Pytorch Lightning which causes losses to spike after resuming training - we're looking into this and will push a fix when we determine what the problem is. For now, if a segmentation training jobs fails it needs to be started from scratch.

### Typical training curves

You can see training metrics on the HuggingFace pages for each model.

In most cases the training loss is correlated with the parameter count of the model (e.g. b5 is better than b0). For validation, the results are a bit less clear and suggest that we are running into the limits of the annotations in the dataset. Larger models perform better, but not by a huge amount; for larger jobs where throughput/latency is important you may be fine running the smallest models. This is also the case in environments where we expect the model to perform very well, like urban canopy coverage.

## Exporting models for production use

TBD

## Benchmarks

Our dataset is relatively unique in the literature in that we provide instance-level masks. Most other datasets either provide tree locations (i.e. single points) or bounding boxes (DeepForest). This makes an independent apples-to-apples evaluation difficult. Cross-validation statistics on our own dataset (see the paper) show that our models perform well and that our labels are probably self-consistent, and we have verified this using partner data - this model is used operationally at Restor.

We provide a few independent benchmarks:

- Detecting tree instances and canopy cover in the city of Zurich using public 10 cm SwissTopo data. The city provides both a high resolution CHM and tree locations for 20k municipal trees.
- Predictions on the NEON Evaluation Dataset; since we do not train on NEON by default we use the training annotations for testing as they are provided as large orthomosaics and this represents a more realistic usage scenario for the models. We compare how our instance-level predictions compare to human annotation in open canopy and how our semantic segmentation models can reconstruct canopy cover.
- Predictions on a large orthomosaic from Tonga with numerous individual trees annotated.
- Qualitative results on some selected images (for example orchard/plantation style sites and a large-scale prediction demo over Swiss aerial imagery).

However we believe that the models speak for themselves - you are encouraged to provide your own imagery to try out. We welcome criticism and feedback if you find that the models aren't predicting as well as you expect.
