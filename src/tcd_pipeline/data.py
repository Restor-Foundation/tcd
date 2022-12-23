import os

import numpy as np
import torch
from rasterio.windows import from_bounds
from torch.utils.data import DataLoader
from torchgeo.datasets import GeoDataset, RasterDataset, stack_samples
from torchgeo.samplers import GridGeoSampler, PreChippedGeoSampler
from torchgeo.samplers.constants import Units

# import kornia as K
# import albumentations as A
# from torchgeo.transforms import AugmentationSequential


def dataloader_from_image(image, tile_size_px, stride_px, gsd_m=0.1, batch_size=1):
    """Yields a torchgeo dataloader from a single (potentially large) image.

    This function is a convenience utility that creates a dataloader for tiled
    inference. Essentially it subclasses RasterDataset with a glob that locates
    a single image (based on the given image path).

    The provided tile size [px] is the square dimension of the input to the model,
    chosen by available VRAM typically. The gsd should similarly be selected as
    appropriate for the model. Together these are used to define what size tile to
    sample from the input image, e.g. tile_size * gsd. We assume that the image
    is in a metric CRS!

    Args:
        image_path (str): Path to image
        tile_size_px (int): Tile size in pixels.
        stride_px (int): Stride, nominally around 1 receptive field
        batch_size (int): Batch size

    Returns:
        _type_: _description_
    """

    # Calculate the sample tile size in metres given
    # the image resolution and the desired GSD

    assert np.allclose(
        image.res[0], gsd_m
    ), f"Image resolution does not match GSD of {gsd_m}m - resize it first."

    height_px, width_px = image.shape
    sample_tile_size = round(min(height_px, width_px, tile_size_px) / 32) * 32
    transforms = None

    """
    if round(image.res[0],3) != round(gsd_m, 3):
        sample_tile_size = tile_size_px * (image.res[0] / gsd_m)
        transforms = AugmentationSequential(
            A.Resize(height=tile_size_px,
            width=tile_size_px), data_keys=["image"]
        )   
    """

    image_path = image.files[0]
    basename = os.path.basename(image_path)
    dirname = os.path.abspath(os.path.dirname(image_path))

    class SingleImageRaster(RasterDataset):
        filename_glob = basename
        is_image = True
        separate_files = False

    dataset = SingleImageRaster(root=dirname, transforms=transforms)

    # If we're exactly the right size, just assume the image is "pre-chipped"
    # this sampler will then return a single geo bounding box.
    if height_px == tile_size_px and width_px == tile_size_px:
        sampler = PreChippedGeoSampler(dataset)

    # Otherwise grid sample as usual
    else:
        sampler = GridGeoSampler(
            dataset, size=sample_tile_size, stride=stride_px, units=Units.PIXELS
        )

    dataloader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, collate_fn=stack_samples
    )

    return dataloader
