# -*- coding: utf-8 -*-
import io
import logging
import os
import pickle
import shutil
import subprocess
from enum import IntEnum
from typing import Optional

import rasterio
import torch
from rasterio.enums import Resampling

logger = logging.getLogger(__name__)

COMPRESSION = "JPEG"


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
            pargs.extend(
                [
                    "-b",
                    "1",
                    "-b",
                    "2",
                    "-b",
                    "3",
                ]
            )

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
            "NUM_THREADS=val/ALL_CPUS",
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
            args.extend([new_path, new_path])
        else:
            base, ext = os.path.splitext(new_path)
            if output_path is None:
                out_path = base + f"_{int(target_gsd_m*100)}" + ext
                args.extend([new_path, out_path])
            else:
                args.extend([new_path, output_path])

        logger.info("Running {}".format(" ".join(args)))
        res = subprocess.check_output(args)
