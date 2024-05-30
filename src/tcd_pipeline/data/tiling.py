import logging
import math
from typing import Generator, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import rasterio
from PIL import Image
from rasterio import windows
from shapely.geometry import box

logger = logging.getLogger(__name__)


def generate_tiles(height, width, tile_size) -> list:
    """
    Generate non-overlapping tile extents covering a source image.
    """

    n_tiles_x = int(math.ceil(width / tile_size))
    n_tiles_y = int(math.ceil(height / tile_size))

    tiles = []

    for tx in range(n_tiles_x):
        for ty in range(n_tiles_y):
            minx = tx * tile_size
            miny = ty * tile_size

            maxx = minx + tile_size
            maxy = miny + tile_size

            tile_box = box(
                minx,
                miny,
                min(maxx, width),
                min(maxy, height),
            )

            tiles.append(tile_box)

    return tiles


class Tiler:
    """
    Helper class to generate tiles over a 2D extent. Can optionally generate tiles with centre weighting,
    but by default returns equally spaced tiles with edges that align with the edges of the image. The
    tiler first determines the minimum number of tiles required to cover an extent and then distributes
    the tiles across it.

    Tiles can be larger than the input size, though in this case you should get a single tile that over-extends
    the array.

    This class returns tile extents and does not have any dependence on the source image or array, end users
    should use TiledImage or TiledGeoImage.
    """

    def __init__(
        self,
        width: int,
        height: int,
        tile_size: int,
        min_overlap: int,
        centre_weight: bool = False,
    ):
        """

        Construct a tiler with the desired output spec (generally a set tile size with a minimum overlap). The
        returned tiles will have at least the minimum overlap; overlap is maximised subject to the number
        of tiles required to cover the image.

        Args:
            width (int): Image width
            height (int): Image height
            tile_size (int): Tile size
            min_overlap (int): Minimum tile overlap
            centre_weight (bool): Distribute tile centres rather than align tile edges

        """
        self.width = width
        self.height = height
        self.tile_size = tile_size
        self.overlap = min_overlap
        self.centre_weight = centre_weight
        self.stride = tile_size - min_overlap

        if self.overlap > tile_size:
            raise ValueError("Overlap must be less than tile size.")

        if (self.width - tile_size) <= 0 and (self.height - tile_size) <= 0:
            self.overlap = 0

    @property
    def tiles(self) -> Generator:
        """
        Returns a generator of tiles (tuples of x, y slices)
        """
        return self._tiles()

    def _n_tiles(self, distance: int) -> int:
        """
        Returns the number of intervals required to cover a distance
        """
        if distance <= self.tile_size:
            return 1

        intervals = math.ceil(
            (distance - self.tile_size) / (self.tile_size - self.overlap)
        )

        return 1 + intervals

    @property
    def effective_overlap(self):
        pass

    def _tile_edges(
        self, extent: int, tile_size: int, stride: int, n_tiles: int
    ) -> List[int]:
        """
        Returns a list of tile edges in ascending axis order (e.g. left -> right)
        """

        if not self.centre_weight:
            return np.linspace(0, extent - tile_size, n_tiles).astype(int)
        else:
            tile_range = min(tile_size, stride) * (n_tiles - 1)
            start = extent - tile_range
            return [(start + i * stride) for i in range(n_tiles)]

    def _tiles(self) -> Generator:
        """
        Internal function for generating tiles. Proceeds roughly as follows:

        1. Determine what stride we need (tile_size - overlap). Stride is the distance
        between tile edges.
        2. Determine how many tiles we need to cover each axis, given a particular overlap
        3. Determine the boundaries of each tile
        4. Lazily generate the tile slices

        Returns:
            tiles: generator of tuple(slice, slice) in xy order
        """

        n_x_tiles = self._n_tiles(self.width)
        n_y_tiles = self._n_tiles(self.height)

        self.x_edges = self._tile_edges(
            self.width, self.tile_size, self.stride, n_x_tiles
        )
        self.y_edges = self._tile_edges(
            self.height, self.tile_size, self.stride, n_y_tiles
        )

        for y in self.y_edges:
            y_start = y
            y_end = y_start + self.tile_size

            for x in self.x_edges:
                x_start = x
                x_end = x_start + self.tile_size

                yield (slice(x_start, x_end, 1), slice(y_start, y_end, 1))


class TiledImage:
    def __init__(
        self, image, tile_size: int = 1024, overlap: int = 256, pad_if_needed=True
    ):
        """
        Helper class to generate tiles of a fixed size from an input image.
        """
        if isinstance(image, str):
            self.image = Image.open(image)
        elif isinstance(image, Image):
            self.image = image
        elif isinstance(image, np.ndarray):
            self.image = Image.fromarray(image)

        self.width = self.image.width
        self.height = self.image.height
        self.array = np.array(self.image)
        self.pad = pad_if_needed

        self.tile_size = tile_size
        self.overlap = overlap

        self.tiler = Tiler(self.width, self.height, self.tile_size, self.overlap)
        self.windows = [w for w in iter(self.tiler.tiles)]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx) -> dict:
        """
        Returns a tile at a specific index. The output is a dictionary containing the following
        keys:
            image: the image data for this tile
            window: tile bounds (slices)
            bbox: bounding box in source image coordinates
        """
        window = self.windows[idx]
        slice_x, slice_y = window

        # Clamp crop to image
        minx = max(0, slice_x.start)
        maxx = min(self.width, slice_x.stop)
        miny = max(0, slice_y.start)
        maxy = min(self.height, slice_y.stop)

        crop = self.array[miny:maxy, minx:maxx, :]

        crop_height, crop_width = crop.shape[:2]

        if self.pad and (crop_width < self.tile_size or crop_height < self.tile_size):
            out = np.zeros(
                (self.tile_size, self.tile_size, self.array.shape[-1]),
                dtype=self.array.dtype,
            )
            offsetx = -slice_x.start if slice_x.start < 0 else 0
            offsety = -slice_y.start if slice_y.start < 0 else 0
            out[offsety : offsety + crop_height, offsetx : offsetx + crop_width] = crop
        else:
            out = crop
            window = (slice(minx, maxx), slice(miny, maxy))

        data = {"image": out, "window": window, "bbox": self._get_bbox(window)}
        return data

    def _get_bbox(self, window) -> box:
        slice_x, slice_y = window

        return box(
            minx=slice_x.start, miny=slice_y.start, maxx=slice_x.stop, maxy=slice_y.stop
        )

    def visualise(
        self, idx=None, midpoints=False, boxes=True, edges=False, image=True
    ) -> None:
        fig, ax = plt.subplots()
        ax.set_aspect("equal")

        ax.imshow(self.array)

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
                sx, sy = w
                xy = (sx.start, sy.start)
                width = sx.stop - sx.start
                height = sy.stop - sy.start
                rect = plt.Rectangle(
                    xy=xy,
                    width=int(width),
                    height=int(height),
                    fill=True,
                    edgecolor="none",
                    alpha=0.3,
                )
                ax.add_patch(rect)

        # Edges

        if edges:
            for w in tiles:
                sx, sy = w
                xy = (sx.start, sy.start)
                width = sx.stop - sx.start
                height = sy.stop - sy.start
                rect = plt.Rectangle(
                    xy=xy,
                    width=width,
                    height=width,
                    fill=False,
                    edgecolor="blue",
                    alpha=0.6,
                )
                ax.add_patch(rect)

        # Centres
        if midpoints:
            for w in tiles:
                sx, sy = w
                width = sx.stop - sx.start
                height = sy.stop - sy.start
                plt.scatter(
                    (sx.start + width / 2),
                    (sy.start + height / 2),
                    marker="+",
                    color="red",
                    alpha=0.5,
                )

        rect = plt.Rectangle(
            xy=(0, 0),
            width=int(self.width),
            height=int(self.height),
            fill=False,
            edgecolor="green",
            alpha=0.5,
        )
        ax.add_patch(rect)
        plt.xlim(-0.1 * self.width, 1.1 * self.width)
        plt.ylim(-0.1 * self.height, 1.1 * self.height)

    def visualise_tile(self, idx, show_valid=False, valid_pad=128) -> None:
        _, ax = plt.subplots()
        tile = self[idx]["image"]
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


class TiledGeoImage(TiledImage):
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

        self.width = self.dataset.width
        self.height = self.dataset.height
        self.pad = pad_if_needed
        self.src_gsd = self.dataset.res[0]
        self.target_gsd = target_gsd

        logger.info(f"Source resolution is {self.src_gsd}")

        if overlap > tile_size:
            raise ValueError(
                "Overlap must be less than tile size to avoid gaps in output."
            )

        # Raw tile size - we will rescale when we return the data
        self.src_tile_size = round(tile_size * self.scale_factor)
        self.tile_size = tile_size
        self.src_overlap = round(overlap * self.scale_factor)

        self.tiler = Tiler(
            self.width, self.height, self.src_tile_size, self.src_overlap
        )
        self.windows = [w for w in iter(self.tiler.tiles)]

    def _get_data(self, idx) -> Tuple[npt.NDArray, Tuple[slice, slice]]:
        """
        Returns unscaled data corresponding to the extent defined by one tile, for
        example if tile_size=1024, src_gsd=0.05 and target_gsd=0.1, this will return
        tiles of size 2048 x 2048.
        """
        window = self.windows[idx]
        slice_x, slice_y = window
        window = windows.Window.from_slices(rows=slice_y, cols=slice_x)

        return (
            self.dataset.read(
                window=window,
                boundless=self.pad,
            ),
            window,
        )

    def _get_bounds(self, window) -> tuple:
        return windows.bounds(window, transform=self.dataset.transform)

    @property
    def scale_factor(self) -> float:
        return round(self.target_gsd / self.dataset.res[0], 6)

    def _get_bbox(self, window) -> box:
        return box(
            minx=window.col_off,
            miny=window.row_off,
            maxx=window.col_off + window.width,
            maxy=window.row_off + window.height,
        )

    def __getitem__(self, idx) -> dict:
        data, window = self._get_data(idx)

        data = data.transpose((1, 2, 0))

        # Blur with "fake" PSF
        if abs(self.scale_factor - 1) > 1e-6:
            kernel_size = int(self.scale_factor * 1.5)

            if kernel_size % 2 or kernel_size == 0:
                kernel_size += 1

            # From CHW
            data = cv2.blur(data, ksize=(kernel_size, kernel_size))
            data = cv2.resize(
                data, (self.tile_size, self.tile_size), interpolation=cv2.INTER_LINEAR
            )

        data = {
            "image": data,
            "bounds": self._get_bounds(window),
            "window": window,
            "scale_factor": self.scale_factor,
            "src_gsd": self.src_gsd,
            "target_gsd": self.target_gsd,
            "bbox": self._get_bbox(window),
        }

        return data

    @property
    def array(self):
        return self.dataset.read().transpose((1, 2, 0))
