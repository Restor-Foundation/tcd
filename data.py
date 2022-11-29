import os

import numpy as np
import rasterio
import torch
from rasterio.windows import from_bounds
from torch.utils.data import DataLoader
from torchgeo.datasets import GeoDataset, stack_samples
from torchgeo.samplers import GridGeoSampler
from torchgeo.samplers.constants import Units


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

    # Calculate desired tile size in metres from desired GSD and
    # tile size in pixels.
    tile_size_m = (tile_size_px * gsd_m, tile_size_px * gsd_m)
    stride_m = stride_px * gsd_m

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
            out_width = round((query.maxx - query.minx) / self.res)
            out_height = round((query.maxy - query.miny) / self.res)

            out_shape = (3, out_height, out_width)
            bounds = (query.minx, query.miny, query.maxx, query.maxy)
            dest = self.image.read(
                out_shape=out_shape,
                window=from_bounds(*bounds, self.image.transform),
            )
            tensor = torch.tensor(dest)
            output = {"image": tensor, "bbox": query, "crs": self._crs}

            return output

        def __len__(self) -> int:
            return 1

    dataset = SingleImageDataset(image)
    sampler = GridGeoSampler(
        dataset, size=tile_size_m, stride=stride_m, units=Units.CRS
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, collate_fn=stack_samples
    )

    return dataloader
