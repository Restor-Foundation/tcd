# -*- coding: utf-8 -*-
import io
import logging
import os
import pickle
import shutil
import subprocess
from enum import IntEnum
from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt
import PIL
import rasterio
import shapely
import shapely.geometry
import torch
from affine import Affine
from PIL import Image
from rasterio import DatasetReader, features
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling, calculate_default_transform, reproject
from rasterio.windows import Window, from_bounds
from rtree import index

logger = logging.getLogger(__name__)

COMPRESSION = "JPEG"

permute_chw_hwc = (1, 2, 0)
permute_hwc_chw = (2, 0, 1)


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

    shape = (int(shape[0]), int(shape[1]))

    return features.rasterize([polygon], out_shape=shape)


class Vegetation(IntEnum):
    """
    Classes of vegetation represented by each number
    """

    CANOPY = 0
    TREE = 1
    CANOPY_SP = 2  # sp = superpixel


def reproject_image(input_path: str, output_path: str, dst_crs: str = "EPSG:3395"):
    with rasterio.Env(GDAL_NUM_THREADS="ALL_CPUS", GDAL_TIFF_INTERNAL_MASK=True) as env:
        with rasterio.open(input_path) as src:
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds
            )

            kwargs = src.meta.copy()
            kwargs.update(
                {
                    "crs": dst_crs,
                    "transform": transform,
                    "width": width,
                    "height": height,
                    "nodata": 0.0,
                    "driver": "GTiff",
                }
            )

            with rasterio.open("reprojected.tif", "w", **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        src_nodata=0,  # Critical to keep masks
                        dst_nodata=0,
                        resampling=Resampling.nearest,
                    )  # For speed here as no resize needed


def resample_image(input_path: str, output_path: str, target_gsd_m: float = 0.1):
    """Resample an image to a target GSD in metres

    Args:
        input_path (str): Path to input image
        output_path (str): Path to output image
        target_gsd_m (float, optional): Target GSD in metres. Defaults to 0.1.

    """

    with rasterio.Env(GDAL_NUM_THREADS="ALL_CPUS", GDAL_TIFF_INTERNAL_MASK=True) as env:
        with rasterio.open(input_path) as src:
            scale, dst_transform = scale_transform(src, target_gsd_m)

            height = int(round(src.height * scale))
            width = int(round(src.width * scale))

            assert scale < 1

            data = src.read(
                out_shape=(src.count, height, width), resampling=Resampling.bilinear
            )

            kwargs = src.meta.copy()
            kwargs.update(
                {
                    "transform": dst_transform,
                    "width": data.shape[-1],
                    "height": data.shape[-2],
                    "nodata": 0,
                }
            )

            with rasterio.open(output_path, "w", **kwargs) as dst:
                dst.write(data)


def image_to_tensor(image: Union[str, torch.Tensor, DatasetReader]) -> torch.Tensor:
    """
    Converts the input into a float tensor in CHW order. If you pass a Tensor in,
    no transpose operation will be performed.
    """
    # Load image if needed
    if isinstance(image, str):
        image = np.array(Image.open(image))
    elif isinstance(image, DatasetReader):
        image = image.read()
    elif not isinstance(image, np.ndarray) and not isinstance(image, torch.Tensor):
        logger.error("Provided image of type %s which is not supported.", type(image))
        raise NotImplementedError

    # Format conversion
    if isinstance(image, torch.Tensor):
        image_tensor = image.float()
    elif isinstance(image, np.ndarray):
        image_tensor = torch.from_numpy(image.transpose((2, 0, 1))).float()

    return image_tensor


def scale_transform(src: rasterio.DatasetReader, target_gsd_m: float):
    """Get the scale factor and scaled transform given a target resolution.

    Input dataset must use a metric CRS.

    Args:

        src (rasterio.DatasetReader): dataset to scale
        target_gsd_m (float): desired GSD

    Returns:

        scale (float): scale factor
        transform (Affine): scaled transform
    """
    t = src.transform

    assert np.allclose(*src.res)

    scale = src.res[0] / target_gsd_m
    transform = Affine(t.a / scale, t.b, t.c, t.d, t.e / scale, t.f)

    return scale, transform


def convert_to_projected(
    path: str,
    output_path: str = None,
    temp_name: Optional[str] = None,
    inplace: Optional[bool] = False,
    resample: Optional[bool] = False,
    target_gsd_m: float = 0.1,
    dst_crs: str = "EPSG:3395",
    use_vrt: bool = True,
) -> None:
    """Convert an input image to projected coordinates and optionally resample

    Args:
        path (str): Path to image (typically a GeoTiff)
        output_path (str, optional): Path to the new stored image
        temp_name (str, optional): Optional temporary filename when processing. Defaults to None.
        inplace (bool, optional): Process input file in place - will overwrite your image! Defaults to False.
        resample (bool, optional): Resample the input image. Defaults to False.
        target_gsd_m (float): Target ground sample distance in metres. Defaults to 0.1.
        use_vrt (bool): Use WarpedVRT instead of GDAL directly, should masking but may be slower

    """

    if use_vrt:
        with rasterio.Env(GDAL_NUM_THREADS="ALL_CPUS", GDAL_TIFF_INTERNAL_MASK=True):
            with rasterio.open(path) as src:
                with WarpedVRT(
                    src,
                    crs=dst_crs,
                    resampling=Resampling.bilinear,
                    warp_mem_limit=256,
                    warp_extras={"NUM_THREADS": "ALL_CPUS"},
                ) as vrt:
                    scale, dst_transform = scale_transform(vrt, target_gsd_m)

                    height = int(round(vrt.height * scale))
                    width = int(round(vrt.width * scale))

                    dst_window = from_bounds(*src.bounds, src.transform)
                    data = vrt.read(
                        window=dst_window, out_shape=(src.count, width, height)
                    )

                profile = src.profile.copy()
                profile.update(
                    {
                        "count": 3,
                        "crs": dst_crs,
                        "transform": dst_transform,
                        "width": width,
                        "height": height,
                        "nodata": 0,
                        "dtype": "uint8",
                        "driver": "GTiff",
                        "compress": "JPEG",
                        "tiled": True,
                        "blockxsize": 512,
                        "blockysize": 512,
                    }
                )

                if output_path is None:
                    base, ext = os.path.splitext(path)
                    output_path = base + f"_proj_{int(target_gsd_m*100)}" + ext

                with rasterio.open(output_path, "w", **profile) as dst:
                    dst.write(data[:3])

                if inplace:
                    shutil.move(output_path, path)

        return

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
            "-r",
            "lanczos",
            "-t_srs",
            dst_crs,
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
            "-r",
            "lanzcos",
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
            "lanczos",
            "-t_srs",
            dst_crs,
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


def format_lat_str(lat) -> str:
    if lat < 0:
        return f"{-lat:.3f}$^\circ$S"
    else:
        return f"{lat:.3f}$^\circ$N"


def format_lon_str(lon) -> str:
    if lon < 0:
        return f"{-lon:.3f}$^\circ$W"
    else:
        return f"{lon:.3f}$^\circ$E"


def paste_array(
    dst: npt.NDArray, src: npt.NDArray, offset: tuple, merge="max"
) -> npt.NDArray:
    """
    Paste src array into dst array at specified offset, handling negative offsets
    and ensuring src does not extend beyond the bounds of dst.

    Args:
        dst (np.ndarray): Destination array where src is to be pasted.
        src (np.ndarray): Source array to be pasted into dst.
        offset (tuple): (xmin, ymin) offset at which to paste src into dst.

    Returns:
        np.ndarray: dst array with src pasted into it.
    """
    xmin, ymin = offset

    src_height, src_width = src.shape
    dst_height, dst_width = dst.shape

    src = src[max(0, -ymin) :, max(0, -xmin) :]

    x_overlap = dst_width - (xmin + src_width)
    if x_overlap < 0:
        src = src[:, :x_overlap]

    y_overlap = dst_height - (ymin + src_height)
    if y_overlap < 0:
        src = src[:y_overlap, :]

    crop_height, crop_width = src.shape

    dst_xmin = max(0, xmin)
    dst_ymin = max(0, ymin)
    dst_xmax = dst_xmin + crop_width
    dst_ymax = dst_ymin + crop_height

    if merge == "max":
        dst[dst_ymin:dst_ymax, dst_xmin:dst_xmax] = np.maximum(
            src, dst[dst_ymin:dst_ymax, dst_xmin:dst_xmax]
        )
    elif merge == "min":
        dst[dst_ymin:dst_ymax, dst_xmin:dst_xmax] = np.minimum(
            src, dst[dst_ymin:dst_ymax, dst_xmin:dst_xmax]
        )
    elif merge == "mean":
        dst[dst_ymin:dst_ymax, dst_xmin:dst_xmax] = (
            src + dst[dst_ymin:dst_ymax, dst_xmin:dst_xmax]
        ) / 2
    else:
        raise NotImplementedError("Currently array merging supports min/max/mean")

    return dst


def find_overlapping_neighbors(
    boxes: shapely.geometry.box,
) -> list[shapely.geometry.box]:
    # Create an R-tree spatial index
    idx = index.Index()

    # Populate the index with bounding boxes of input boxes
    for i, b in enumerate(boxes):
        idx.insert(i, b.bounds)

    # Dictionary to store overlapping neighbors for each box
    overlapping_neighbors = {}

    # Iterate over each box and find its overlapping neighbors
    for i, b in enumerate(boxes):
        # Find overlapping candidates using the spatial index
        potential_neighbors = idx.intersection(b.bounds)

        # Filter potential neighbors to those that actually overlap
        overlapping_neighbors[i] = [
            j for j in potential_neighbors if boxes[i].intersects(boxes[j]) and i != j
        ]

    return overlapping_neighbors


def inset_box(box, pad):
    minx, miny, maxx, maxy = box.bounds
    return shapely.geometry.box(minx + pad, miny + pad, maxx - pad, maxy - pad)
