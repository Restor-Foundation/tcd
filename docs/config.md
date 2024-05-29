# Pipeline configuration

The default model configurations should work well in most situations. We provide configs for instance and semantic segmentation using the [Hydra](https://hydra.cc/) framework from Facebook. We host our released models on Huggingface Hub for ease of download and in most cases the weights will automatically be pulled for you.

## Customising the config

To override settings, you can use the `overrides` parameter in `Pipeline`, for example:

```python
pipeline = Pipeline('semantic', overrides=[f"model.weights=restor/tcd-segformer-mit-b5",
                                              "model.config.model=segformer",
                                              "model.batch_size=2",
                                              "data.tile_size=1024"])
```

You should adjust `tile_size` depending on how much VRAM your graphics card has. If you have something powerful like a RTX3090 with a large amount of VRAM, you can probably perform inference with tiles up to 2048px for most models. The overlap should be at least 256-512 to minimise edge-effects when tiling and stitching. This corresponds to approximately a single receptive field for ResNet50 but theory doesn't always work well in practice so feel free to tweak this. The higher the overlap the more tiles you'll need to predict. For more intuition and understanding, you can look at the tiling notebook.

It isn't necessarily a good idea to choose huge tile sizes, because some models (like Segformer) predict masks with a fixed resolution and smaller details will be lost. So even if your GPU _can_ predict an image at 4096x4096, it is probably better to predict at 1024 or 2048 and relying on tiling. For semantic segmentation, the pipeline can operate in batched mode so the prediction time should be roughly proportional to the number of pixels you're trying to predict (e.g. if you can predict at 2048x2048 with batch size 2, you can probably predict at 1024x1024 with batch size 16).

The pipeline will automatically attempt to use CUDA, but if it's not available, it will fallback to either `cpu` or `mps` if available.

You should not need to modify any other parameters unless you're trying to train new models.

## Configuration Groups

TODO