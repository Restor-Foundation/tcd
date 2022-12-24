import os

import numpy as np
import rasterio
import torch
from rasterio.windows import Window, from_bounds
from torch.utils.data import DataLoader
from torchgeo.datasets import GeoDataset, stack_samples
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

    if isinstance(image, str):
        image = rasterio.open(image)

    # Calculate the sample tile size in metres given
    # the image resolution and the desired GSD

    assert np.allclose(
        image.res[0], gsd_m
    ), f"Image resolution does not match GSD of {gsd_m}m - resize it first."

    height_px, width_px = image.shape
    sample_tile_size = round(min(height_px, width_px, tile_size_px) / 32) * 32

    class SingleImageDataset(GeoDataset):
        def __init__(self, image, transforms=None) -> None:
            self.image = image
            self._crs = image.crs
            self.res = image.res[0]
            self.coords = (
                image.bounds.left,
                image.bounds.right,
                image.bounds.bottom,
                image.bounds.top,
                0,
                np.infty,
            )
            super().__init__(transforms)
            self.index.insert(0, self.coords, "")

        def __getitem__(self, query):
            out_shape = (self.image.count, sample_tile_size, sample_tile_size)
            bounds = (query.minx, query.miny, query.maxx, query.maxy)

            dest = self.image.read(
                out_shape=out_shape, window=from_bounds(*bounds, self.image.transform)
            )

            tensor = torch.tensor(dest)
            output = {"image": tensor, "bbox": query, "crs": self._crs}

            return output

        def __len__(self) -> int:
            return 1

    dataset = SingleImageDataset(image)

    # If we're exactly the right size, just assume the image is "pre-chipped"
    # this sampler will then return a single geo bounding box.
    if height_px == sample_tile_size and width_px == sample_tile_size:
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
