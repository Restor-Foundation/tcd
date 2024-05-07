import logging
from typing import Union

import rasterio
import rasterio.windows
from torch.utils.data import DataLoader, Dataset

from .tiling import TiledGeoImage, TiledImage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
Dataloaders for tiling orthomosaic imagery.


"""


def collate_dicts(samples):
    collated_dict = {}
    for dictionary in samples:
        for key in dictionary:
            collated_dict.setdefault(key, []).append(dictionary.get(key))

    return {
        key: None if all(value is None for value in values) else values
        for key, values in collated_dict.items()
    }


class SingleImageDataset(Dataset):
    def __init__(
        self, image, tile_size: int = 1024, overlap: int = 256, pad_if_needed=True
    ):
        self.tiles = TiledImage(
            image, tile_size=tile_size, overlap=overlap, pad_if_needed=pad_if_needed
        )

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        return self.tiles[idx]


class SingleImageGeoDataset(SingleImageDataset):
    def __init__(
        self,
        image,
        target_gsd: float = 0.1,
        tile_size: int = 1024,
        overlap: int = 256,
        pad_if_needed: bool = True,
    ):
        self.image = image
        self.tiles = TiledGeoImage(
            image,
            target_gsd=target_gsd,
            tile_size=tile_size,
            overlap=overlap,
            pad_if_needed=pad_if_needed,
        )


def dataloader_from_image(
    image: Union[str, rasterio.DatasetReader],
    tile_size_px: int = 1024,
    overlap_px: int = 256,
    gsd_m: float = 0.1,
    batch_size: int = 1,
    pad_if_needed: bool = True,
) -> DataLoader:
    """Yields a Pytorch dataloader from a single (potentially large) image.

    This function is a convenience utility that creates a dataloader for tiled
    inference.

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
        pad_if_needed (bool): Pad to the specified tile size, defaults to True
    Returns:
        DataLoader: torch dataloader for this image
    """
    assert tile_size_px % 32 == 0

    if isinstance(image, str):
        image = rasterio.open(image)

    if image.res[0] != 0:
        logger.info("Geographic information present, loading as a geo dataset")
        dataset = SingleImageGeoDataset(
            image,
            target_gsd=gsd_m,
            tile_size=tile_size_px,
            overlap=overlap_px,
            pad_if_needed=pad_if_needed,
        )
    else:
        logger.warn(
            "Unable to determine GSD/resolution, loading as a plain image dataset"
        )
        dataset = SingleImageDataset(
            image,
            tile_size=tile_size_px,
            overlap=overlap_px,
            pad_if_needed=pad_if_needed,
        )

    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_dicts)

    return dataloader
