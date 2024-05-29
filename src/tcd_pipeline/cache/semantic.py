import glob
import gzip
import logging
import math
import os
import pickle
from collections import defaultdict
from typing import List

import numpy as np
import numpy.typing as npt
import rasterio
import rasterio.windows
import rtree
from rasterio.windows import Window
from shapely.geometry import box

from .cache import ResultsCache

logger = logging.getLogger(__name__)


class SemanticSegmentationCache(ResultsCache):
    @property
    def results(self) -> List[dict]:
        """
        A list of ProcessedInstances that were stored in the cache folder
        """
        return self._results


class PickleSemanticCache(SemanticSegmentationCache):
    def _find_cache_files(self) -> List[str]:
        return glob.glob(os.path.join(self.cache_folder, f"*_{self.cache_suffix}.pkl"))

    def save(self, mask, bbox: box):
        output = {"mask": mask, "bbox": bbox, "image": self.image_path}

        file_name = f"{self.tile_count}_{self.cache_suffix}.pkl"
        output_path = os.path.join(self.cache_folder, file_name)

        with open(output_path, "wb") as fp:
            pickle.dump(output, fp)

        self.tile_count += 1

    def _load_file(self, cache_file: str) -> dict:
        """Load pickled cache results

        Args:
            cache_file (str): Cache filename

        Returns:
            dict: dictionary containing "instances" and "bbox"

        """

        with open(cache_file, "rb") as fp:
            annotations = pickle.load(fp)

        return annotations


class NumpySemanticCache(SemanticSegmentationCache):
    def _find_cache_files(self) -> List[str]:
        return glob.glob(os.path.join(self.cache_folder, f"*_{self.cache_suffix}.npz"))

    def save(self, mask: npt.NDArray, bbox: box):
        """
        Save cached results in Numpy format.

        Args:
            mask: prediction array
            bbox: tile bounding box in global image coordinates
        """
        file_name = f"{self.tile_count}_{self.cache_suffix}.npz"
        output_path = os.path.join(self.cache_folder, file_name)

        np.savez_compressed(
            output_path,
            mask=mask,
            bbox=bbox.bounds,
            image=self.image_path,
            tile_count=self.tile_count,
        )

        self.tile_count += 1

    def _load_file(self, cache_file: str) -> dict:
        """Load cached results from MS-COCO format

        Args:
            cache_file (str): Cache filename

        Returns:
            dict

        """

        res = np.load(cache_file)

        out = {}
        out["bbox"] = box(*res["bbox"])
        out["mask"] = res["mask"]
        out["image"] = str(res["image"])
        out["tile_id"] = self.tile_count

        return out


class GeotiffSemanticCache(SemanticSegmentationCache):
    def __init__(
        self, cache_tile_size: int = 10000, compress: bool = False, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        with rasterio.open(self.image_path) as src:
            self.image_width = src.width
            self.image_height = src.height
            self.src_transform = src.transform
            self.src_meta = src.meta

        # Must be larger/equal to tile size
        self.cache_tile_size = min(
            self.image_height, min(self.image_width, cache_tile_size)
        )
        self.compress = compress

        if self.classes is not None:
            self.band_count = max(1, len(self.classes) - 1)
        else:
            self.band_count = 1

        self.tile_names = []
        self.tile_prediction_count = None
        self.intersections = None
        self._generate_tiles()

    def _generate_tiles(self):
        """
        Internal function that generates non-overlapping tile extents covering the source image.

        Tiles are stored as bounding boxes in an R-Tree so that when predictions are saved to the
        cache, we can efficiently look up which cache tiles overlap and where the tile data should
        be saved. The tiler here is not fancy - it figures out the number of tiles required to cover
        each axis and generates bounding boxes for each.
        """

        n_tiles_x = int(math.ceil(self.image_width / self.cache_tile_size))
        n_tiles_y = int(math.ceil(self.image_height / self.cache_tile_size))

        self.index = rtree.Index()

        idx = 0
        for tx in range(n_tiles_x):
            for ty in range(n_tiles_y):
                minx = tx * self.cache_tile_size
                miny = ty * self.cache_tile_size

                maxx = minx + self.cache_tile_size
                maxy = miny + self.cache_tile_size

                tile_box = box(
                    minx,
                    miny,
                    min(maxx, self.image_width),
                    min(maxy, self.image_height),
                )
                self.index.add(id=idx, coordinates=tile_box.bounds, obj=tile_box)

                idx += 1

        logger.debug(
            f"Caching to {idx} tiles, approximately {(idx * self.cache_tile_size ** 2 )/1e9 :1.2f}GB needed for temporary storage during inference"
        )

    def set_prediction_tiles(self, dataset_tiles):
        """
        Pre-compute intersections for dataset tiles which allows for compression
        operations to happen during prediction. This can keep working storage space
        much lower than waiting for the prediction to complete before compressing.

        By calculating how many hits each cache tile is expected to have, we can
        run compression when the number of hits (or in this case a counter reaching
        zero) has happened.
        """

        self.tile_prediction_count = defaultdict(int)
        self.intersections = defaultdict(list)
        for tile in dataset_tiles:
            for result in self.index.intersection(tile.bounds, objects=True):
                self.tile_prediction_count[result.id] += 1
                self.intersections[tile.bounds].append(result)

    def _create_cache_tile(self, tile_bbox: box, path: str) -> None:
        """
        Create an empty tile with a given bounding box. The tile is generated with reference
        to the source image being used for prediction and it will have the same CRS and a transform
        derived from the source and the desired bounding box.

        By default, no compression is enabled to speed up cache time during inference.
        """
        meta = self.src_meta
        meta.update(
            {
                "driver": "GTiff",
                "width": self.cache_tile_size,
                "height": self.cache_tile_size,
                "count": self.band_count,
                "dtype": "uint8",
                "nodata": 0,
                "compress": "deflate" if self.compress else None,
                "transform": rasterio.windows.transform(
                    Window(*tile_bbox.bounds), self.src_transform
                ),
            }
        )

        self.tile_names.append(os.path.basename(path))

        with rasterio.open(path, "w+", **meta) as dst:
            # Touch a very small window to create the tile
            dst.write(
                np.zeros((1, 1), dtype=np.uint8), window=Window(0, 0, 1, 1), indexes=1
            )

    def _find_cache_files(self) -> List[str]:
        """
        Returns a list of cached GeoTIFFs (i.e. saved tiles)
        """
        return [os.path.join(self.cache_folder, n) for n in self.tile_names]

    def save_tile(self, mask: npt.NDArray, bbox: box):
        """
        Save a model prediction to the tile cache. This function will determine which
        tiles in the cache overlap with the provided bounding box and the predictions
        will be split up appropriately. Cache tiles are created lazily - i.e. as needed
        so it is possible that not all the tiles in the index will be created if there are
        large regions of empty data in the input image.

        Args:
            mask: prediction result
            bbox: tile bounding box in global image pixel coordinates

        """

        if self.intersections is not None:
            intersecting_cache_tiles = self.intersections[bbox.bounds]
        else:
            intersecting_cache_tiles = [
                hit for hit in self.index.intersection(bbox.bounds, objects=True)
            ]

        for tile in intersecting_cache_tiles:
            tile_idx = tile.id
            tile = tile.object
            minx, miny, maxx, maxy = [int(i) for i in bbox.intersection(tile).bounds]

            # Crop size
            width = int(maxx - minx)
            height = int(maxy - miny)

            # Coordinates from mask
            mask_offset_x = int(minx - bbox.bounds[0])
            mask_offset_y = int(miny - bbox.bounds[1])
            mask_crop = mask[
                :,
                mask_offset_y : mask_offset_y + height,
                mask_offset_x : mask_offset_x + width,
            ]

            # Coordinates within tile
            tile_offset_x = minx - tile.bounds[0]
            tile_offset_y = miny - tile.bounds[1]

            window = Window(tile_offset_x, tile_offset_y, width, height)

            file_name = f"{tile_idx}{self.cache_suffix}.tif"
            output_path = os.path.join(self.cache_folder, file_name)

            if not os.path.exists(output_path):
                self._create_cache_tile(tile, output_path)

            with rasterio.open(output_path, "r+", nodata=0, dtype="uint8") as dst:
                for band_idx, band in enumerate(mask_crop[1:, :, :]):
                    dst.write(band, indexes=band_idx + 1, window=window)

            if self.tile_prediction_count is not None:
                self.tile_prediction_count[tile_idx] -= 1
                if self.tile_prediction_count[tile_idx] == 0:
                    self.compress_tile(output_path)

    def save(self, mask: npt.NDArray, bbox: box):
        """
        Save a prediction mask into the cache. See `save_tile` for
        more information on the internal details.

        The provided mask should be an unsigned 8-bit array containing prediction
        values scaled from 0 to 255. Nominally, 0 is used as the nodata
        value in the cache tile. If the maximum value of the array is
        greater than 1, then the array is multiplied by 255 and cast
        to uint8.

        The mask can have multiple bands, corresponding to multiple class
        predictions, but the first channel is assumed to be background and is
        not stored (as it can be reconstructed from the remaining bands).

        Args:
            mask: prediction result
            bbox: tile bounding box in global image coordinates
        """

        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, 0)

        if mask.max() <= 1:
            mask = (mask * 255).astype(np.uint8)

        self.save_tile(mask, bbox)

    def compress_tile(self, path):
        with rasterio.open(path) as src:
            meta = src.meta
            meta.update({"driver": "GTiff", "compress": "deflate"})

            with rasterio.open(path + ".temp", "w", **meta) as dst:
                dst.write(src.read())

        import shutil

        logger.debug("Compressing", path)
        shutil.move(path + ".temp", path)

    def compress_tiles(self):
        """
        Iterate over the tiles in the cache and re-write them as compressed
        GeoTIFFs. This usually results in a significant reduction in file
        size. The `deflate` compression method is used as `packbits` can sometimes
        fail, and `lzw` is often not supported and is slow.

        A temporary file is created before being moved to overwrite the source image.
        """
        for path in self._find_cache_files():
            self.compress_tile(path)

    def generate_vrt(self, filename="overview.vrt", root=None):
        """
        Generate a virtual raster from the tiles in the cache. This should be called
        at the end of inference to create an "overview" file that can be used to
        read all the tiles as a single image.
        """
        if root is None:
            root = self.cache_folder

        import subprocess

        _ = subprocess.check_output(
            [
                "gdalbuildvrt",
                "-srcnodata",
                "0",
                "-vrtnodata",
                "0",
                filename,
                *self.tile_names,
            ],
            cwd=root,
        )

    def _load_file(self, cache_file: str) -> rasterio.DatasetReader:
        return rasterio.open(cache_file)
