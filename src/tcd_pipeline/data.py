import logging
import os
from typing import Union

import kornia.augmentation as K
import numpy as np
import rasterio
import torch
from rasterio.windows import Window, from_bounds
from torch.utils.data import DataLoader
from torchgeo.datasets import GeoDataset, stack_samples
from torchgeo.samplers import GridGeoSampler, PreChippedGeoSampler
from torchgeo.samplers.constants import Units
from torchgeo.transforms import AugmentationSequential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def dataloader_from_image(
    image: Union[str, rasterio.DatasetReader],
    tile_size_px: int,
    stride_px: int,
    gsd_m: float = 0.1,
    batch_size: int = 1,
    pad_if_needed: bool = False,
):
    """Yields a torchgeo dataloader from a single (potentially large) image.

    This function is a convenience utility that creates a dataloader for tiled
    inference. Internally this is done by creating a subclass of GeoDataset. The
    image resolution will be checked against the provided GSD and gsd_m will be
    used, to avoid rounding errors when sampling.

    Don't forget the stride is not the overlap. So you should calculate the stride
    based on (tile_size - overlap) to be around 1 receptive field.

    The provided tile size [px] is the square dimension of the input to the model,
    chosen by available VRAM typically. The gsd should similarly be selected as
    appropriate for the model. Together these are used to define what size tile to
    sample from the input image, e.g. tile_size * gsd. We assume that the image
    is in a metric CRS!

    Args:
        image_path (str or DatasetReader): Path to image
        tile_size_px (int): Tile size in pixels.
        stride_px (int): Stride in pixels
        gsd_m (float): Assumed GSD, defaults to 0.1
        batch_size (int): Batch size, defaults to 1
        pad_if_needed (bool): Pad to the specified tile size, defaults to False
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

            # Bit of a hack here to avoid rounding
            # - we already checked that the input
            # has the correct GSD, so this clobber
            # is kind of OK
            self.res = gsd_m
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

            output = {"image": tensor.float(), "bbox": query, "crs": self._crs}

            if self.transforms is not None:
                output = self.transforms(output)

            return output

        def __len__(self) -> int:
            return 1

    if pad_if_needed:
        transforms = AugmentationSequential(
            K.PadTo([tile_size_px, tile_size_px], keepdim=True), data_keys=["image"]
        )
    else:
        transforms = None

    dataset = SingleImageDataset(image, transforms=transforms)

    # If we're exactly the right size, just assume the image is "pre-chipped"
    # this sampler will then return a single geo bounding box.
    if height_px == sample_tile_size and width_px == sample_tile_size:
        sampler = PreChippedGeoSampler(dataset)

    # Otherwise grid sample as usual
    else:
        sampler = GridGeoSampler(
            dataset, size=sample_tile_size, stride=stride_px, units=Units.PIXELS
        )

        if logger.isEnabledFor(logging.DEBUG):
            """
            Sanity check sampler vs image
            """
            logger.debug(f"Sampler hits: {sampler.length}")
            logger.debug(f"Image resolution: {image.res}")
            logger.debug(f"Image bounds: {image.bounds}")
            logger.debug(f"Sampler size: {sampler.size}, stride: {sampler.stride}")

            for hit in sampler.hits:
                logger.debug(f"Hit bounds: {hit.bounds}")

                import math

                from torchgeo.datasets import BoundingBox

                bounds = BoundingBox(*hit.bounds)

                rows = (
                    math.ceil(
                        (bounds.maxy - bounds.miny - sampler.size[0])
                        / sampler.stride[0]
                    )
                    + 1
                )
                cols = (
                    math.ceil(
                        (bounds.maxx - bounds.minx - sampler.size[1])
                        / sampler.stride[1]
                    )
                    + 1
                )

                logger.debug(f"Sampler rows/cols: {(rows, cols)}")

    dataloader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, collate_fn=stack_samples
    )

    return dataloader
