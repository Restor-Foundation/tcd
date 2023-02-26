# Notes on model conversion:

The `convert_torchscript.py` script contains demo code to convert our Detectron2 instance segmentation model to torchscript allowing for (hopefully) dependency-free usage. You can run this script as follows:

```bash
 python convert_torchscript.py --config-file ../config/detectron2/detectron_mask_rcnn.yaml --output ./output --export-method tracing --format torchscript --run-eval MODEL.WEIGHTS 
../checkpoints/model_final.pth MODEL.DEVICE cpu
```

Note that currently the number of classes is hardcoded into the convert file (todo). This assumes you have the datasets set up as per the repository instructions (otherwise you need to register the datasets 
yourself).
