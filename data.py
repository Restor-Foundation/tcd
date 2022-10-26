import os

from torch.utils.data import DataLoader, Dataset
from torchgeo.datasets import RasterDataset, stack_samples
from torchgeo.samplers import GridGeoSampler
from torchgeo.samplers.constants import Units


def dataloader_from_image(image_path, tile_size_px, stride_px, gsd_m=0.1, batch_size=1):
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
    filename = os.path.basename(image_path)
    dirname = os.path.dirname(image_path)

    # Calculate desired tile size in metres from desired GSD and
    # tile size in pixels.
    tile_size_m = (tile_size_px * gsd_m, tile_size_px * gsd_m)
    stride_m = stride_px * gsd_m

    class SingleImageRaster(RasterDataset):
        filename_glob = filename
        is_image = True
        separate_files = False

    dataset = SingleImageRaster(root=dirname)
    sampler = GridGeoSampler(
        dataset, size=tile_size_m, stride=stride_m, units=Units.CRS
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, collate_fn=stack_samples
    )

    return dataloader
