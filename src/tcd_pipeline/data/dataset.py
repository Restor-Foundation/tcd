import logging
import math
import os
from typing import Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.windows
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..util import Bbox

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


class SingleImageGeoDataset(Dataset):
    def __init__(
        self,
        image,
        target_gsd: float = 0.1,
        tile_size: int = 1024,
        overlap: int = 256,
        pad_if_needed: bool = True,
    ):
        if isinstance(image, str):
            self.dataset = rasterio.open(image)
        elif isinstance(image, rasterio.DatasetReader):
            self.dataset = image
        else:
            raise NotImplementedError(f"Input type {type(image)} not supported.")

        self.pad = pad_if_needed
        self.src_gsd = self.dataset.res[0]
        self.target_gsd = target_gsd
        # self.clip_tiles = clip_tiles

        if overlap > tile_size:
            raise ValueError(
                "Overlap must be less than tile size to avoid gaps in output."
            )

        self.tile_size = tile_size
        self.overlap = overlap

        self.tile_extent = self.target_gsd * self.tile_size

        x_same = np.allclose(
            0, self.dataset.bounds.right - self.dataset.bounds.left - self.tile_extent
        )
        y_same = np.allclose(
            0, self.dataset.bounds.top - self.dataset.bounds.bottom - self.tile_extent
        )
        if x_same and y_same:
            self.overlap_extent = 0
        else:
            self.overlap_extent = self.target_gsd * self.overlap

        self.width = self.dataset.width
        self.height = self.dataset.height
        self.windows = [b for b in iter(self._windows())]

    def __len__(self):
        return len(self.windows)

    def _get_data(self, idx):
        window = self.windows[idx]
        return (
            self.dataset.read(
                window=window,
                boundless=self.pad,
            ),
            window,
        )

    def _get_extent(self, window):
        return rasterio.windows.bounds(window, transform=self.dataset.transform)

    @property
    def scale_factor(self):
        return round(self.target_gsd / self.dataset.res[0], 6)

    def _get_bbox(self, window):
        return Bbox(
            minx=window.col_off,
            miny=window.row_off,
            maxx=window.col_off + window.width,
            maxy=window.row_off + window.height,
        )

    def __getitem__(self, idx):
        data, window = self._get_data(idx)

        # Blur with "fake" PSF
        scale_factor = int(self.target_gsd / self.dataset.res[0])
        if scale_factor != 1:
            kernel_size = int(scale_factor * 1.5)

            if kernel_size % 2:
                kernel_size += 1

            # From CHW
            data = cv2.blur(data.transpose((1, 2, 0)), ksize=(kernel_size, kernel_size))
            data = cv2.resize(
                data, (self.tile_size, self.tile_size), interpolation=cv2.INTER_LINEAR
            )
            # Back to CHW
            data = data.transpose((2, 0, 1))

        # Create dict with necessary information
        data = {
            "image": torch.from_numpy(data),
            "extent": self._get_extent(window),
            "window": window,
            "scale_factor": self.scale_factor,
            "src_gsd": self.src_gsd,
            "target_gsd": self.target_gsd,
            "bbox": self._get_bbox(window),
        }

        return data

    def visualise(self, idx=None, midpoints=False, boxes=True, edges=False, image=True):
        fig, ax = plt.subplots()
        ax.set_aspect("equal")

        ax.imshow(self.dataset.read().transpose((1, 2, 0)))

        if idx is not None:
            tiles = []
            if isinstance(idx, int):
                idx = [idx]
            for i in idx:
                tiles.append(self.windows[i])
        else:
            tiles = self.windows

        # Tile fill
        if boxes:
            for w in tiles:
                rect = plt.Rectangle(
                    xy=(w.col_off, w.row_off),
                    width=int(w.width),
                    height=int(w.height),
                    fill=True,
                    edgecolor="none",
                    alpha=0.3,
                )
                ax.add_patch(rect)

        # Edges

        if edges:
            for w in tiles:
                rect = plt.Rectangle(
                    xy=(w.col_off, w.row_off),
                    width=int(w.width),
                    height=int(w.height),
                    fill=False,
                    edgecolor="blue",
                    alpha=0.6,
                )
                ax.add_patch(rect)

        # Centres
        if midpoints:
            for w in tiles:
                plt.scatter(
                    (w.col_off + w.width / 2),
                    (w.row_off + w.height / 2),
                    marker="+",
                    color="red",
                    alpha=0.5,
                )

        rect = plt.Rectangle(
            xy=(0, 0),
            width=int(self.dataset.width),
            height=int(self.dataset.height),
            fill=False,
            edgecolor="green",
            alpha=0.5,
        )
        ax.add_patch(rect)
        plt.xlim(-0.1 * self.dataset.width, 1.1 * self.dataset.width)
        plt.ylim(-0.1 * self.dataset.height, 1.1 * self.dataset.height)

    def visualise_tile(self, idx, show_valid=False, valid_pad=128):
        _, ax = plt.subplots()
        tile = self[idx]["image"].permute(1, 2, 0)
        ax.imshow(tile)

        if show_valid:
            pad = valid_pad
            tile_height, tile_width = tile.shape[:2]
            rect = plt.Rectangle(
                (pad, pad),
                width=tile_width - 2 * pad,
                height=tile_height - 2 * pad,
                alpha=0.5,
                color="green",
            )
            ax.add_patch(rect)

    def _tile_midpoints(self, extent, tile, stride, n_tiles):
        midpoint_range = min(tile, stride) * (n_tiles - 1)
        start = (extent - midpoint_range) / 2
        return [(start + i * stride) for i in range(n_tiles)]

    def _n_windows(self, range_size, width, overlap):
        subrange_count = 0
        covered_size = 0
        offset = 0

        if range_size == width:
            return 1

        while covered_size < range_size:
            covered_size = offset + width
            offset = covered_size - overlap
            subrange_count += 1

        return subrange_count

    def _windows(self):
        stride = self.tile_extent - self.overlap_extent

        image_extent_x = self.width * self.dataset.res[0]
        image_extent_y = self.height * self.dataset.res[1]

        n_x_windows = self._n_windows(
            image_extent_x, self.tile_extent, self.overlap_extent
        )
        n_y_windows = self._n_windows(
            image_extent_y, self.tile_extent, self.overlap_extent
        )

        x_midpoints = self._tile_midpoints(
            image_extent_x, self.tile_extent, stride, n_x_windows
        )
        y_midpoints = self._tile_midpoints(
            image_extent_y, self.tile_extent, stride, n_y_windows
        )

        for y in y_midpoints:
            y_start = self.dataset.bounds.bottom + y - self.tile_extent / 2
            y_end = y_start + self.tile_extent

            for x in x_midpoints:
                x_start = self.dataset.bounds.left + x - self.tile_extent / 2
                x_end = x_start + self.tile_extent

                """
                if self.clip_tiles:
                    x_start = max(self.dataset.bounds.left, x_start)
                    x_end = min(self.dataset.bounds.right, x_end)
                    y_start = max(self.dataset.bounds.bottom, y_start)
                    y_end = min(self.dataset.bounds.top, y_end)
                """

                window = rasterio.windows.from_bounds(
                    left=x_start,
                    right=x_end,
                    bottom=y_start,
                    top=y_end,
                    transform=self.dataset.transform,
                )

                yield window


class SingleImageDataset(SingleImageGeoDataset):
    def __init__(
        self, image, tile_size: int = 1024, overlap: int = 256, pad_if_needed=True
    ):
        if isinstance(image, str):
            self.dataset = Image.open(image)
        elif isinstance(image, Image):
            self.dataset = image
        elif isinstance(image, np.ndarray):
            self.dataset = Image.fromarray(image)

        self.width = self.dataset.width
        self.height = self.dataset.height
        self.image = np.array(self.dataset)
        self.pad = pad_if_needed

        self.src_gsd = None
        self.target_gsd = None

        if overlap > tile_size:
            raise ValueError(
                "Overlap must be less than tile size to avoid gaps in output."
            )

        self.tile_size = tile_size
        self.overlap = overlap

        if np.allclose(0, self.dataset.width - tile_size) and np.allclose(
            0, self.dataset.height - tile_size
        ):
            self.overlap = 0

        self.windows = [b for b in iter(self._windows())]

    def _get_data(self, idx):
        window = self.windows[idx]
        return torch.Tensor(self.image[window].transpose((2, 0, 1))), window

    def _get_extent(self, window):
        return None

    def _get_bbox(self, window):
        slice_y, slice_x = window

        return Bbox(
            minx=slice_x.start, miny=slice_y.start, maxx=slice_x.stop, maxy=slice_y.stop
        )

    def _windows(self):
        stride = self.tile_size - self.overlap

        n_x_windows = self._n_windows(self.width, self.tile_size, self.overlap)
        n_y_windows = self._n_windows(self.height, self.tile_size, self.overlap)

        x_midpoints = self._tile_midpoints(
            self.width, self.tile_size, stride, n_x_windows
        )
        y_midpoints = self._tile_midpoints(
            self.height, self.tile_size, stride, n_y_windows
        )

        for y in y_midpoints:
            y_start = y - self.tile_size / 2
            y_end = y_start + self.tile_size

            for x in x_midpoints:
                x_start = x - self.tile_size / 2
                x_end = x_start + self.tile_size

                yield (slice(x_start, x_end), slice(y_start, y_end))


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
