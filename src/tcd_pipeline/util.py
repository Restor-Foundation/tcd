# -*- coding: utf-8 -*-
import io
import logging
import os
import pickle
import shutil
import subprocess
from enum import IntEnum
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
import rasterio
import shapely
import torch
from rasterio import features
from rasterio.windows import Window

logger = logging.getLogger(__name__)

COMPRESSION = "JPEG"


class Bbox:
    """A bounding box with integer coordinates."""

    def __init__(self, minx: float, miny: float, maxx: float, maxy: float) -> None:
        """Initializes the Bounding box

        Args:
            minx (float): minimum x coordinate of the box
            miny (float): minimum y coordinate of the box
            maxx (float): maximum x coordiante of the box
            maxy (float): maximum y coordinate of the box
        """
        self.minx = (int)(minx)
        self.miny = (int)(miny)
        self.maxx = (int)(maxx)
        self.maxy = (int)(maxy)
        self.bbox = (self.minx, self.miny, self.maxx, self.maxy)
        self.width = self.maxx - self.minx
        self.height = self.maxy - self.miny
        self.area = self.width * self.height

    def overlap(self, other: Any) -> bool:
        """Checks whether this bbox overlaps with another one

        Args:
            other (Bbox): other bbox

        Returns:
            bool: Whether or not the bboxes overlap
        """
        if (
            self.minx >= other.maxx
            or self.maxx <= other.minx
            or self.maxy <= other.miny
            or self.miny >= other.maxy
        ):
            return False
        return True

    @classmethod
    def from_array(self, a):
        return self(*a)

    def __array__(self) -> npt.NDArray:
        return np.array([self.minx, self.miny, self.maxx, self.maxy])

    def window(self) -> Window:
        return Window(self.minx, self.miny, self.width, self.height)

    def __str__(self) -> str:
        return f"Bbox(minx={self.minx:.4f}, miny={self.miny:.4f}, maxx={self.maxx:.4f}, maxy={self.maxy:.4f})"


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


def load_results_from_pickle(filename: str):
    if torch.cuda.is_available():
        with open("./model_pickles/" + filename + ".pickle", "rb") as handle:
            results = pickle.load(handle)
    else:
        with open("./model_pickles/" + filename + ".pickle", "rb") as handle:
            results = CPU_Unpickler(handle).load()

    return results


def get_pickle_result(pickle_path: str):
    """
    Read any pickle file given path

    Args:
                    pickle_path (str): path to a pickle file

    Returns:
                    read result
    """
    if torch.cuda.is_available():
        with open(pickle_path, "rb") as handle:
            results = pickle.load(handle)
    else:
        with open(pickle_path, "rb") as handle:
            results = CPU_Unpickler(handle).load()
    return results


def mask_to_polygon(mask: npt.NDArray[np.bool_]) -> shapely.geometry.MultiPolygon:
    """Converts the mask of an object to a MultiPolygon

    Args:
        mask (np.array(bool)): Boolean mask of the segmented object

    Returns:
        MultiPolygon: Shapely MultiPolygon describing the object
    """

    all_polygons = []
    for shape, _ in features.shapes(mask.astype(np.int16), mask=mask):
        all_polygons.append(shapely.geometry.shape(shape))

    all_polygons = shapely.geometry.MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        # need to keep it a Multi throughout
        if all_polygons.type == "Polygon":
            all_polygons = shapely.geometry.MultiPolygon([all_polygons])
    return all_polygons


def polygon_to_mask(
    polygon: shapely.geometry.Polygon, shape: tuple[int, int]
) -> npt.NDArray:
    """Rasterise a polygon to a mask

    Args:
        polygon: Shapely Polygon describing the object
    Returns:
        np.array(np.bool_): Boolean mask of the segmented object
    """

    return features.rasterize([polygon], out_shape=shape)


class Vegetation(IntEnum):
    """
    Classes of vegetation represented by each number
    """

    CANOPY = 0
    TREE = 1
    CANOPY_SP = 2  # sp = superpixel


def convert_to_projected(
    path: str,
    output_path: str = None,
    temp_name: Optional[str] = None,
    inplace: Optional[bool] = False,
    resample: Optional[bool] = False,
    target_gsd_m: float = 0.1,
) -> None:
    """Convert an input image to projected coordinates and optionally resample

    Args:
        path (str): Path to image (typically a GeoTiff)
        output_path (str, optional): Path to the new stored image
        temp_name (str, optional): Optional temporary filename when processing. Defaults to None.
        inplace (bool, optional): Process input file in place - will overwrite your image! Defaults to False.
        resample (bool, optional): Resample the input image. Defaults to False.
        target_gsd_m (float): Target ground sample distance in metres. Defaults to 0.1.

    """
    with rasterio.open(path) as img:

        working_dir = os.path.dirname(path)
        filename, ext = os.path.splitext(os.path.basename(path))

        if temp_name is None:
            temp_name = "_" + filename

        temporary_vrt = os.path.join(working_dir, f"{temp_name}_m.vrt")

        # Convert to a VRT for speed
        logger.info(f"Converting {temp_name} to projected CRS")
        pargs = [
            "gdalwarp",
            "-multi",
            "-wo",
            "NUM_THREADS=ALL_CPUS",
            "-co",
            "TILED=YES",
            "-co",
            "BLOCKXSIZE=512",
            "-co",
            "BLOCKYSIZE=512",
            "-co",
            "COMPRESS=NONE",
            "-co",
            "SRC_METHOD=NO_GEOTRANSFORM",
            "-t_srs",
            "EPSG:3395",
            "-ot",
            "Byte",
            "-overwrite",
            "-of",
            "vrt",
            f"{path}",
            temporary_vrt,
        ]

        try:
            subprocess.check_output(pargs)
        except subprocess.CalledProcessError as error:
            logger.error(error.output)

        temporary_tif = os.path.join(working_dir, f"{temp_name}_m.tif")

        # Then compress
        logger.info(f"Compressing {temp_name}")
        pargs = [
            "gdal_translate",
            "-co",
            "NUM_THREADS=ALL_CPUS",
            "-co",
            "BIGTIFF=IF_SAFER",
            "-co",
            f"COMPRESS=JPEG",
            "-co",
            "TILED=YES",
            "-co",
            "BLOCKXSIZE=512",
            "-co",
            "BLOCKYSIZE=512",
        ]

        if img.count != 1:
            pargs.extend(["-co", "PHOTOMETRIC=YCBCR"])

        if img.count > 3:
            pargs.extend(["-b", "1", "-b", "2", "-b", "3", "-mask", "4"])

        pargs.append(temporary_vrt)
        pargs.append(temporary_tif)

        try:
            subprocess.check_output(pargs)
        except subprocess.CalledProcessError as error:
            logger.error(error.output)

        if inplace:
            shutil.move(temporary_tif, path)
            os.remove(temporary_vrt)
            new_path = path
        else:
            new_path = os.path.join(working_dir, f"{filename}_proj{ext}")
            shutil.move(temporary_tif, new_path)
            os.remove(temporary_vrt)

    if resample:
        args = [
            "gdalwarp",
            "-multi",
            "-wo",
            "NUM_THREADS=ALL_CPUS",
            "-co",
            "BIGTIFF=IF_SAFER",
            "-co",
            f"COMPRESS=JPEG",
            "-co",
            "TILED=YES",
            "-co",
            "BLOCKXSIZE=512",
            "-co",
            "BLOCKYSIZE=512",
            "-r",
            "bilinear",
            "-t_srs",
            "EPSG:3395",
            "-tr",
            f"{target_gsd_m}",
            f"{target_gsd_m}",
            "-overwrite",
        ]

        if inplace:
            args.append("-overwrite")

        base, ext = os.path.splitext(new_path)

        if output_path is None:
            output_path = base + f"_{int(target_gsd_m*100)}" + ext

        args.extend([new_path, output_path])

        logger.info("Running {}".format(" ".join(args)))
        res = subprocess.check_output(args)

        if inplace:
            shutil.move(output_path, path)
