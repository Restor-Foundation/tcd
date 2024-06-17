# Model Zoo

Below you can find links to models on HuggingFace and performance metrics on our holdout dataset. If you want to use one in a pipeline, just run:

```python
from tcd_pipeline import Pipeline

pipeline = Pipeline(<model name>)
pipeline.predict(<your image>)
```
and the model type should be automagically determined. Weights will be downloaded/cached if you don't have them on your system already.

## Contributing

If you train a model with our dataset and would like to integrate it into the pipeline, please let us know and we'd be happy to add it to the zoo.

## Instance Segmentation

We currently provide a trained Mask-RCNN model with a ResNet50 backbone - in the future we will provide alternative backbone sizes and hopefully some newer architectures.

| Model Architecture  | Model Tag | mAP50  |
| ------------------  | --------- | --------- |
| Mask-RCNN Resnet34  | `restor/tcd-mask-rcnn-r34` | TBD |
| Mask-RCNN Resnet50  | [`restor/tcd-mask-rcnn-r50`](https://huggingface.co/restor/tcd-mask-rcnn-r50) | 43.22 |
| Mask-RCNN Resnet101 | `restor/tcd-mask-rcnn-r101` | TBD |

## Semantic segmentation

| Model Architecture | Model Tag | Accuracy  | F1       | IoU     |
| ------------------ | --------- | --------- | -------- | ------- |
| U-Net ResNet34      | [`restor/tcd-unet-r34`](https://huggingface.co/restor/tcd-unet-r34)         | 0.883 | 0.871 | 0.838 |
| U-Net ResNet50      | [`restor/tcd-unet-r50`](https://huggingface.co/restor/tcd-unet-r50)         | 0.881 | 0.880 | 0.849 |
| U-Net ResNet101     | [`restor/tcd-unet-r101`](https://huggingface.co/restor/tcd-unet-r101)         | **0.900** | 0.886 | 0.856 |
| Segformer mit-b0   | [`restor/tcd-segformer-mit-b0`](https://huggingface.co/restor/tcd-segformer-mit-b0) | 0.892 | 0.882 | 0.865 |
| Segformer mit-b1   | [`restor/tcd-segformer-mit-b1`](https://huggingface.co/restor/tcd-segformer-mit-b1) | 0.897 | 0.891 | 0.870 |
| Segformer mit-b2   | [`restor/tcd-segformer-mit-b2`](https://huggingface.co/restor/tcd-segformer-mit-b2) | 0.889 | 0.898 | 0.871 |
| Segformer mit-b3   | [`restor/tcd-segformer-mit-b3`](https://huggingface.co/restor/tcd-segformer-mit-b3) | 0.884 | 0.901 | 0.875 |
| Segformer mit-b4   | [`restor/tcd-segformer-mit-b4`](https://huggingface.co/restor/tcd-segformer-mit-b4) | 0.891 | 0.901 | 0.875 |
| Segformer mit-b5   | [`restor/tcd-segformer-mit-b5`](https://huggingface.co/restor/tcd-segformer-mit-b5) | 0.890 | **0.902** | **0.876** |