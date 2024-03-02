import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt
import psutil
import rasterio
import rasterio.crs
import rasterio.windows
import shapely
import shapely.geometry
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class ProcessedResult(ABC):
    def set_threshold(self, new_threshold: int) -> None:
        """Sets the threshold of the ProcessedResult, also regenerates
        prediction masks

        Args:
            new_threshold (double): new confidence threshold
        """
        self.confidence_threshold = new_threshold
        self._generate_masks()

    def get_hardware_information(self):
        """Returns the hardware information of the system

        Returns:
            dict: hardware information
        """

        self.hardware = {}

        self.hardware["physical_cores"] = psutil.cpu_count()
        self.hardware["logical_cores"] = psutil.cpu_count(logical=True)
        self.hardware["system_memory"] = psutil.virtual_memory().total

        try:
            self.hardware["cpu_frequency"] = psutil.cpu_freq().max
        except:
            pass

        if torch.cuda.is_available():
            self.hardware["gpu_model"]: torch.cuda.get_device_name(0)
            self.hardware["gpu_memory"]: torch.cuda.get_device_properties(
                0
            ).total_memory

        return self.hardware

    @abstractmethod
    def visualise(self):
        pass

    @abstractmethod
    def serialise(self):
        pass

    @abstractmethod
    def _generate_masks(self):
        pass

    @abstractmethod
    def load_serialisation(self):
        pass

    def filter_geometry(self, geometry):
        pass

    def save_shapefile(*args, **kwargs):
        raise NotImplementedError

    def set_roi(
        self,
        shape: Union[dict, shapely.geometry.Polygon],
        crs: Optional[rasterio.crs.CRS] = None,
    ):
        """Filter by geometry, should be a simple Polygon

        Args:
            shape_dict (dict): shape

        """

        if crs is not None and crs != self.image.crs:
            logger.warning("Geometry CRS is not the same as the image CRS, warping")
            shape = rasterio.warp.transform_geom(crs, self.image.crs, shape)

        if not isinstance(shape, shapely.geometry.Polygon):
            shape = shapely.geometry.shape(shape)

        if not isinstance(shape, shapely.geometry.Polygon):
            logger.warning("Input shape is not a polygon, not applying filter")
            return

        self.valid_region = shapely.geometry.polygon.orient(shape)
        self._filter_roi()
        self._generate_masks()

    def _filter_roi(self):
        logger.warning("No filter function defined, so filtering by ROI has no effect")

    @property
    def num_valid_pixels(self) -> int:
        if self.valid_region is not None:
            return int(self.valid_region.area / (self.image.res[0] ** 2))
        else:
            return np.count_nonzero(self.image.read().mean(0) > 0)

    @property
    def canopy_cover(self) -> float:
        return np.count_nonzero(self.canopy_mask) / self.num_valid_pixels

    def _save_mask(self, mask: npt.NDArray, output_path: str, binary=True):
        """Saves a mask array to a GeoTiff file

        Args:
            mask (np.array): mask to save
            output_path (str): path to save mask to
            image_path (str): path to source image, default None
            binary (bool): save a binary mask or not, default True

        """

        if self.image is not None:
            if self.valid_mask is not None:
                mask *= self.valid_mask
                out_transform = rasterio.windows.transform(
                    self.valid_window, self.image.transform
                )
            else:
                out_transform = self.image.transform

            out_meta = self.image.meta
            out_height = mask.shape[0]
            out_width = mask.shape[1]

            out_crs = self.image.crs
            out_meta.update(
                {
                    "driver": "GTiff",
                    "height": out_height,
                    "width": out_width,
                    "compress": "packbits",
                    "count": 1,
                    "nodata": 0,
                    "transform": out_transform,
                    "crs": out_crs,
                }
            )

            with rasterio.env.Env(GDAL_TIFF_INTERNAL_MASK=True):
                with rasterio.open(
                    output_path,
                    "w",
                    nbits=1 if binary else 8,
                    **out_meta,
                ) as dest:
                    dest.write(mask, indexes=1, masked=True)
        else:
            logger.warning(
                "No base image provided, output masks will not be georeferenced"
            )
            Image.fromarray(mask).save(output_path, compress="packbits")
