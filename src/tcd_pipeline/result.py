import json
import logging
import os
import time
from abc import ABC, abstractmethod
from glob import glob
from typing import Any, Optional, Union

import fiona
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import psutil
import pycocotools.coco
import rasterio
import rasterio.crs
import rasterio.windows
import seaborn as sns
import shapely
import shapely.geometry
import torch
import yaml
from PIL import Image
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from skimage.transform import resize
from tqdm.auto import tqdm

from .instance import ProcessedInstance, dump_instances_coco
from .util import Bbox, Vegetation, format_lat_str, format_lon_str

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

        self.valid_region = shape
        self._filter_roi()
        self._generate_masks()

    def _filter_roi(self):
        logger.debug("No filter function defined, so filtering by ROI has no effect")

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
            if self.valid_region is not None:
                mask, out_transform = rasterio.mask.mask(
                    self.image,
                    [self.valid_region],
                    crop=False,
                    nodata=0,
                    filled=True,
                    invert=False,
                )
            else:
                out_transform = self.image.transform

            out_meta = self.image.meta
            out_height = self.image.height
            out_width = self.image.width

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


class InstanceSegmentationResult(ProcessedResult):
    """A processed result of a model. It contains all trees separately and also a global tree mask, canopy mask and image"""

    def __init__(
        self,
        image: rasterio.DatasetReader,
        instances: Optional[list] = [],
        confidence_threshold: float = 0.2,
        config: dict = None,
    ) -> None:
        """Initializes the Processed Result

        Args:
            image (rasterio.DatasetReader): source image that instances are referenced to
            instances (List[ProcessedInstance], optional): list of all instances. Defaults to []].
            confidence_threshold (int): confidence threshold for retrieving instances. Defaults to 0.2
        """
        self.image = image
        self.instances = instances
        self.valid_region = None
        self.prediction_time_s = -1
        self.config = config
        self.set_threshold(confidence_threshold)

    def _filter_roi(self):
        if self.valid_region is not None:
            self.instances = [
                instance
                for instance in self.instances
                if instance.transformed_polygon(self.image.transform).within(
                    self.valid_region
                )
            ]

        else:
            logger.warning("Unable to filter instances as no ROI has been set.")

    def get_instances(self, only_labeled=False) -> list:
        """Gets the instances that have at score above the threshold

        Returns:
            List[ProcessedInstance]: List of processed instances, all classes
            only_labeled (bool): whether or not to only return labeled instances
        """
        if not only_labeled:
            return [
                instance
                for instance in self.instances
                if instance.score >= self.confidence_threshold
            ]
        else:
            return [
                instance
                for instance in self.instances
                if instance.score >= self.confidence_threshold
                and instance.label is not None
            ]

    def get_trees(self, only_labeled=False) -> list:
        """Gets the trees with a score above the threshold

        Returns:
            List[ProcessedInstance]: List of trees
            only_labeled (bool): whether or not to only return labeled instances
        """
        if not only_labeled:
            return [
                instance
                for instance in self.instances
                if instance.score >= self.confidence_threshold
                and instance.class_index == Vegetation.TREE
            ]
        else:
            return [
                instance
                for instance in self.instances
                if instance.score >= self.confidence_threshold
                and instance.label is not None
                and instance.class_index == Vegetation.TREE
            ]

    def visualise(
        self,
        color_trees: Optional[tuple[int, int, int]] = (255, 105, 180),
        color_canopy: Optional[tuple[int, int, int]] = (255, 243, 0),
        alpha: Optional[float] = 0.5,
        output_path: Optional[str] = None,
        labels: Optional[bool] = False,
        max_pixels: Optional[tuple[int, int]] = None,
        **kwargs: Optional[Any],
    ) -> None:
        """Visualizes the result

        Args:
            color_trees (tuple, optional): rgb value of the trees. Defaults to (204, 0, 0).
            color_canopy (tuple, optional): rgb value of the canopy. Defaults to (0, 0, 204).
            alpha (float, optional): alpha value. Defaults to 0.3.
            output_path (str, optional): if provided, save image instead of showing it
            max_pixels (tuple, optional): max pixel size of output image (memory optimization)
            labels (bool, optional): whether or not to show the labels.
        """
        fig, ax = plt.subplots(**kwargs)
        plt.axis("off")

        tree_mask = self.tree_mask
        canopy_mask = self.canopy_mask
        window = rasterio.windows.from_bounds(
            *self.image.bounds, transform=self.image.transform
        )

        if self.valid_region is not None:
            _, _, window = rasterio.mask.raster_geometry_mask(
                self.image, [self.valid_region], crop=True
            )

        tree_mask = self.tree_mask[window.toslices()]
        canopy_mask = self.canopy_mask[window.toslices()]

        reshape_factor = 1
        if max_pixels is not None:
            reshape_factor = min(
                max_pixels[0] / window.height, max_pixels[1] / window.width
            )
            reshape_factor = min(reshape_factor, 1)

        shape = (
            int(window.height * reshape_factor),
            int(window.width * reshape_factor),
        )

        vis_image = self.image.read(
            out_shape=(self.image.count, shape[0], shape[1]),
            resampling=Resampling.bilinear,
            masked=True,
            window=window,
        ).transpose(1, 2, 0)

        ax.imshow(vis_image)

        resized_canopy_mask = canopy_mask
        if reshape_factor < 1:
            resized_canopy_mask = resize(self.canopy_mask, shape)

        canopy_mask_image = np.zeros((*resized_canopy_mask.shape, 4), dtype=np.uint8)
        canopy_mask_image[resized_canopy_mask > 0] = list(color_canopy) + [255]
        ax.imshow(canopy_mask_image, alpha=alpha)

        resized_tree_mask = tree_mask
        if reshape_factor < 1:
            resized_tree_mask = resize(tree_mask, shape)

        tree_mask_image = np.zeros((*resized_tree_mask.shape, 4), dtype=np.uint8)
        tree_mask_image[resized_tree_mask > 0] = list(color_trees) + [255]

        from skimage import measure

        contours = measure.find_contours(resized_tree_mask, 0.5)

        ax.imshow(tree_mask_image, alpha=alpha)
        for contour in contours:
            ax.plot(
                contour[:, 1],
                contour[:, 0],
                linewidth=0.3,
                color=[c / 255.0 for c in color_trees],
                alpha=min(1, 1.4 * alpha),
            )

        if labels:
            x = []
            y = []
            c = []

            colors = sns.color_palette("bright", 10)
            for tree in self.get_trees():
                coords_poly = tree.polygon.centroid.coords[0]
                coords = [coords_poly[1], coords_poly[0]]

                if tree.label is not None:
                    x.append(coords[1] * reshape_factor)
                    y.append(coords[0] * reshape_factor)
                    c.append(colors[tree.label])

            ax.scatter(x=x, y=y, color=c, s=4)

        plt.tight_layout()

        if output_path is not None:
            plt.savefig(output_path)
        else:
            plt.show()

    def serialise(
        self,
        output_folder: str,
        overwrite: bool = True,
        file_prefix: Optional[str] = "results",
    ) -> dict:
        """Serialise results to a COCO JSON file.

        Args:
            output_folder (str): output folder
            overwrite (bool, optional): overwrite existing data, defaults True
            image_path (str): path to source image, default None
            file_prefix (str, optional): file name, defaults to results
        """

        logger.info(f"Serialising results to {output_folder}/{file_prefix}.json")
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, f"{file_prefix}.json")

        if os.path.exists(output_path) and not overwrite:
            logger.error(
                f"Output file already exists {output_path}, will not overwrite."
            )

        categories = {
            Vegetation.TREE: Vegetation.TREE.name.lower(),
            Vegetation.CANOPY: Vegetation.CANOPY.name.lower(),
        }

        meta = {}
        meta["threshold"] = self.confidence_threshold
        meta["prediction_time_s"] = self.prediction_time_s
        meta["config"] = self.config
        meta["hardware"] = self.get_hardware_information()

        if isinstance(self.config.model.config, str):
            with open(self.config.model.config) as fp:
                meta["config"]["model"]["config"] = yaml.safe_load(fp)
        elif isinstance(self.config.model.config, dict):
            meta["config"]["model"]["config"] = self.config.model.config
        else:
            raise NotImplementedError(
                f"Unknown model config type {self.config.model.config}"
            )

        return dump_instances_coco(
            output_path,
            instances=self.instances,
            image_path=self.image.name,
            categories=categories,
            metadata=meta,
        )

    @classmethod
    def load_serialisation(
        cls,
        input_file: str,
        image_path: Optional[str] = None,
        use_basename: Optional[bool] = True,
        global_mask: Optional[bool] = False,
    ):
        """Loads a ProcessedResult based on a COCO formatted json serialization file. This is useful
        if you want to load in another dataset that uses COCO formatting, or for example if you want
        to load results from a single image. The json file must have an 'images' entry. If you don't
        provide a path then we assume that you want all the results.

        Args:
            input_file (str): serialised instances as COCO-formatted JSON file
            image_path (str, optional): Path where the image is stored. Defaults to the location mentioned in the output_file.
            use_basename (bool, optional): Use basename of image to query file, defaults True
        Returns:
            ProcessedResult: ProcessedResult described by the file
        """
        instances = []

        reader = pycocotools.coco.COCO(input_file)

        if image_path is None:
            image_id = 0
        else:
            query = os.path.basename(image_path) if use_basename else image_path

            image_id = None
            if len(reader.dataset["images"]) == 1:
                image_id = reader.dataset["images"][0]["id"]
            else:
                for img in reader.dataset["images"]:
                    if img["file_name"] == query:
                        image_id = img["id"]

        ann_ids = reader.getAnnIds([image_id])

        if len(ann_ids) == 0:
            logger.warning("No annotations found with this image ID.")

        image = rasterio.open(image_path)

        for ann_id in tqdm(ann_ids):
            annotation = reader.anns[ann_id]
            instance = ProcessedInstance.from_coco_dict(
                annotation, image.shape, global_mask
            )
            if np.count_nonzero(instance.local_mask) != 0:
                instances.append(instance)

        res = cls(
            image,
            instances,
            confidence_threshold=reader.dataset["metadata"].get("threshold", 0),
            config=reader.dataset["metadata"].get("config", {}),
        )
        res.prediction_time_s = reader.dataset["metadata"].get("prediction_time_s", -1)

        return res

    def _generate_masks(self):
        self.canopy_mask = self._generate_mask(Vegetation.CANOPY)
        self.tree_mask = self._generate_mask(Vegetation.TREE)

    def _generate_mask(self, class_id: Vegetation) -> npt.NDArray:
        """Generates a global mask for the given class_id

        Args:
            class_id (Vegetation): Class ID for the mask to be generated

        Returns:
            np.array: mask

        """

        mask = np.full((self.image.height, self.image.width), fill_value=False)

        for instance in self.get_instances():
            if instance.class_index == class_id:

                mask[
                    instance.bbox.miny : instance.bbox.maxy,
                    instance.bbox.minx : instance.bbox.maxx,
                ] |= (
                    instance.local_mask != 0
                )

        return mask

    def save_masks(
        self,
        output_path: str,
        suffix: Optional[str] = "",
        prefix: Optional[str] = "",
    ) -> None:
        """Save prediction masks for tree and canopy. If a source image is provided
        then it is used for georeferencing the output masks.

        Args:
            output_path (str): folder to store data
            suffix (str, optional): mask filename suffix
            prefix (str, optional): mask filename prefix

        """

        os.makedirs(output_path, exist_ok=True)

        self._save_mask(
            mask=self.tree_mask,
            output_path=os.path.join(output_path, f"{prefix}tree_mask{suffix}.tif"),
        )
        self._save_mask(
            mask=self.canopy_mask,
            output_path=os.path.join(output_path, f"{prefix}canopy_mask{suffix}.tif"),
        )

    def save_shapefile(self, output_path: str, indices: Vegetation = None) -> None:
        """Save instances to a georeferenced shapefile.

        Args:
            output_path (str): output file path
            class_index (Vegetation, optional): on
        """

        schema = {
            "geometry": "MultiPolygon",
            "properties": {"score": "float", "class": "str"},
        }

        with fiona.open(
            output_path, "w", "ESRI Shapefile", schema=schema, crs=self.image.crs.wkt
        ) as layer:
            for instance in self.get_instances():

                if indices is not None and instance.class_index not in indices:
                    continue

                elem = {}

                world_polygon = instance.transformed_polygon(self.image.transform)

                if isinstance(instance.polygon, shapely.geometry.Polygon):
                    polygon = shapely.geometry.MultiPolygon([world_polygon])
                else:
                    polygon = world_polygon

                elem["geometry"] = shapely.geometry.mapping(polygon)
                elem["properties"] = {
                    "score": instance.score,
                    "class": "tree"
                    if instance.class_index == Vegetation.TREE
                    else "canopy",
                }

                layer.write(elem)

    @property
    def tree_cover(self):
        return np.count_nonzero(self.tree_mask) / self.num_valid_pixels

    def __str__(self) -> str:
        """String representation, returns canopy and tree cover for image."""
        return (
            f"ProcessedResult(n_trees={len(self.get_trees())},"
            f" canopy_cover={self.canopy_cover:.4f}, tree_cover={self.tree_cover:.4f})"
        )


class SegmentationResult(ProcessedResult):
    def __init__(
        self,
        image: rasterio.DatasetReader,
        tiled_masks: Optional[list] = [],
        bboxes: list[Bbox] = [],
        confidence_threshold: float = 0.2,
        merge_pad: int = 128,
        config: dict = None,
    ) -> None:

        self.image = image
        self.masks = tiled_masks
        self.bboxes = bboxes
        self.merge_pad = merge_pad
        self.valid_region = None
        self.prediction_time_s = -1
        self.config = config

        self.set_threshold(confidence_threshold)

    def serialise(
        self,
        output_folder: str,
        overwrite: bool = True,
        file_prefix: Optional[str] = "results",
    ) -> None:
        """Serialise raw prediction masks. Masks are stored as NPZ files with the
        keys "mask" and "bbox" as well as a timestamp which can be used as a sanity
        check when loading. A JSON file containing a list of masks will also be created.

        Args:
            output_folder (str): output folder
            overwrite (bool, optional): overwrite existing data, defaults True
            file_prefix (str, optional): file name, defaults to results
        """

        logger.info(f"Serialising results to {output_folder}/{file_prefix}")
        os.makedirs(output_folder, exist_ok=True)

        meta_path = os.path.join(output_folder, f"{file_prefix}.json")

        if os.path.exists(meta_path) and not overwrite:
            logger.error(
                f"Output metadata already exists {meta_path}, will not overwrite."
            )
            return

        timestamp = time.time()
        metadata = {}
        metadata["image"] = self.image.name
        metadata["timestamp"] = int(timestamp)
        metadata["masks"] = []
        metadata["confidence_threshold"] = self.confidence_threshold
        metadata["prediction_time_s"] = self.prediction_time_s
        metadata["config"] = self.config
        metadata["hardware"] = self.get_hardware_information()

        if isinstance(self.config.model.config, str):
            with open(self.config.model.config) as fp:
                metadata["config"]["model"]["config"] = yaml.safe_load(fp)
        elif isinstance(self.config.model.config, dict):
            metadata["config"]["model"]["config"] = self.config.model.config
        else:
            raise NotImplementedError(
                f"Unknown model config type {self.config.model.config}"
            )

        for i, item in enumerate(zip(self.masks, self.bboxes)):
            mask, bbox = item
            file_name = f"{file_prefix}_{i}.npz"
            output_path = os.path.join(output_folder, file_name)

            if os.path.exists(output_path) and not overwrite:
                logger.error(
                    f"Output file already exists {output_path}, will not overwrite."
                )
                return
            np.savez_compressed(
                file=output_path,
                mask=mask[0][0],
                image_bbox=mask[0][1],
                bbox=np.array(bbox),
                timestamp=int(timestamp),
            )
            metadata["masks"].append(os.path.abspath(output_path))

        with open(meta_path, "w") as fp:
            json.dump(metadata, fp, indent=1)

    @classmethod
    def load_serialisation(cls, input_file: str, image_path: Optional[str] = None):
        """Loads a ProcessedResult based on a COCO formatted json serialization file. This is useful
        if you want to load in another dataset that uses COCO formatting, or for example if you want
        to load results from a single image. The json file must have an 'images' entry. If you don't
        provide a path then we assume that you want all the results.

        Args:
            input_file (str): serialised instance metadata JSON file
            image_path (Optional[str]): image path, optional
        Returns:
            SegmentationResult: SegmentationResult described by the file
        """
        tiled_masks = []
        bboxes = []

        with open(input_file, "r") as fp:
            metadata = json.load(fp)

        image_path = image_path if image_path else metadata["image"]
        image = rasterio.open(image_path)

        for mask_file in metadata["masks"]:
            data = np.load(mask_file, allow_pickle=True)

            tiled_masks.append([[data["mask"], data["image_bbox"]]])
            bboxes.append(Bbox.from_array(data["bbox"]))

            if data["timestamp"] != metadata["timestamp"]:
                logger.error(
                    "Timestamp in mask and metadat file don't match. Corrupted export?"
                )

        res = cls(
            image=image,
            tiled_masks=tiled_masks,
            bboxes=bboxes,
            confidence_threshold=metadata["confidence_threshold"],
            config=metadata["config"],
        )

        res.prediction_time_s = metadata["prediction_time_s"]

        return res

    def set_threshold(self, new_threshold: int) -> None:
        """Sets the threshold of the ProcessedResult, also regenerates
                prediction masks
        ?
                Args:
                    new_threshold (double): new confidence threshold
        """
        self.confidence_threshold = new_threshold
        self._generate_masks()

    def save_masks(
        self,
        output_path: str,
        suffix: Optional[str] = "",
        prefix: Optional[str] = "",
        pad=0,
    ) -> None:
        """Save prediction masks for tree and canopy. If a source image is provided
        then it is used for georeferencing the output masks.

        Args:
            output_path (str): folder to store data
            image_path (str, optional): source image
            suffix (str, optional): mask filename suffix
            prefix (str, optional): mask filename prefix

        """

        os.makedirs(output_path, exist_ok=True)

        canopy_mask = np.array(self.canopy_mask)

        if pad > 0:
            canopy_mask[:, :pad] = 0
            canopy_mask[:pad, :] = 0
            canopy_mask[:, -pad:] = 0
            canopy_mask[-pad:, :] = 0

        self._save_mask(
            mask=canopy_mask,
            output_path=os.path.join(output_path, f"{prefix}canopy_mask{suffix}.tif"),
        )

        confidence_mask = np.array((255 * self.confidence_map)).astype(np.uint8)

        if pad > 0:
            confidence_mask[:, :pad] = 0
            confidence_mask[:pad, :] = 0
            confidence_mask[:, -pad:] = 0
            confidence_mask[-pad:, :] = 0

        self._save_mask(
            mask=confidence_mask,
            output_path=os.path.join(
                output_path, f"{prefix}canopy_confidence{suffix}.tif"
            ),
            binary=False,
        )

    def _generate_masks(self, average=True) -> npt.NDArray:
        """
        Merges segmentation masks following the strategy outlined in:
        https://arxiv.org/ftp/arxiv/papers/1805/1805.12219.pdf

        1) We clip masks by a fixed amount before merging, this limits
        the effect of edge effects on the final mask.

        2) We merge masks by taking the average value at each overlap

        """

        pad = self.merge_pad
        self.canopy_mask = np.zeros(self.image.shape, dtype=bool)
        self.confidence_map = np.zeros(self.image.shape)

        p = torch.nn.Softmax2d()

        for i, bbox in enumerate(self.bboxes):

            mask, image_bbox = self.masks[i][0]

            confidence = p(torch.Tensor(mask))

            # pred = torch.argmax(confidence, dim=0).numpy()
            _, height, width = confidence.shape

            pad_slice = (
                slice(pad, min(height, bbox.height) - pad),
                slice(pad, min(width, bbox.width) - pad),
            )

            print(bbox.minx, bbox.maxx, bbox.miny, bbox.maxy)

            # TODO check appropriate merge strategy
            if average:
                self.confidence_map[bbox.miny : bbox.maxy, bbox.minx : bbox.maxx][
                    pad_slice
                ] = np.maximum(
                    self.confidence_map[bbox.miny : bbox.maxy, bbox.minx : bbox.maxx][
                        pad_slice
                    ],
                    confidence[1][pad_slice],
                )
            else:
                self.confidence_map[bbox.miny : bbox.maxy, bbox.minx : bbox.maxx][
                    pad_slice
                ] = confidence[1][pad_slice]

        self.canopy_mask = self.confidence_map > self.confidence_threshold

        return

    def visualise(
        self,
        dpi=400,
        max_pixels: Optional[tuple[int, int]] = None,
        output_path=None,
        color_trees: Optional[tuple[int, int, int]] = (255, 105, 180),
        color_canopy: Optional[tuple[int, int, int]] = (0, 0, 204),
        alpha: Optional[float] = 0.5,
        **kwargs,
    ) -> None:
        """Visualise the results of the segmentation. If output path is not provided, the results
        will be displayed.

        Args:
            dpi (int, optional): dpi of the output image. Defaults to 200.
            max_pixels: maximum image size
            output_path (str, optional): path to save the output plots. Defaults to None.
            **kwargs: remaining arguments passed to figure creation

        """

        window = rasterio.windows.from_bounds(
            *self.image.bounds, transform=self.image.transform
        )

        if self.valid_region is not None:
            _, _, window = rasterio.mask.raster_geometry_mask(
                self.image, [self.valid_region], crop=True
            )

        confidence_map = self.confidence_map[window.toslices()]

        reshape_factor = 1
        if max_pixels is not None:
            reshape_factor = min(
                max_pixels[0] / window.height, max_pixels[1] / window.width
            )
            reshape_factor = min(reshape_factor, 1)

        shape = (
            int(window.height * reshape_factor),
            int(window.width * reshape_factor),
        )

        vis_image = self.image.read(
            out_shape=(self.image.count, shape[0], shape[1]),
            resampling=Resampling.bilinear,
            masked=True,
            window=window,
        ).transpose(1, 2, 0)

        resized_confidence_map = confidence_map
        if reshape_factor < 1:
            resized_confidence_map = resize(confidence_map, shape)

        # Normal figure
        fig = plt.figure(dpi=dpi, **kwargs)
        ax = plt.axes()
        ax.tick_params(axis="both", which="major", labelsize="x-small")
        ax.tick_params(axis="both", which="minor", labelsize="xx-small")

        latlon_bounds = transform_bounds(
            self.image.crs,
            "EPSG:4236",
            *rasterio.windows.bounds(window, self.image.transform),
        )
        ax.imshow(
            vis_image,
            extent=[
                latlon_bounds[0],
                latlon_bounds[2],
                latlon_bounds[1],
                latlon_bounds[3],
            ],
        )

        # ax.set_xticks(ax.get_xticks()[::2])
        # ax.set_yticks(ax.get_yticks()[::2])

        ax.set_xticklabels([format_lon_str(x) for x in ax.get_xticks()], rotation=45)
        ax.set_yticklabels([format_lat_str(y) for y in ax.get_yticks()], rotation=45)

        if output_path is not None:
            plt.savefig(os.path.join(output_path, "raw_image.jpg"), bbox_inches="tight")

        # Canopy Mask
        fig = plt.figure(dpi=dpi, **kwargs)
        ax = plt.axes()
        ax.tick_params(axis="both", which="major", labelsize="x-small")
        ax.tick_params(axis="both", which="minor", labelsize="xx-small")
        ax.imshow(vis_image)

        confidence_mask_image = np.zeros(
            (*resized_confidence_map.shape, 4), dtype=np.uint8
        )
        confidence_mask_image[
            resized_confidence_map > self.confidence_threshold
        ] = list(color_trees) + [255]
        ax.imshow(confidence_mask_image, alpha=alpha)

        if output_path is not None:
            plt.savefig(
                os.path.join(output_path, "canopy_overlay.jpg"), bbox_inches="tight"
            )

        # Confidence Map
        fig = plt.figure(dpi=dpi, **kwargs)
        ax = plt.axes()
        ax.tick_params(axis="both", which="major", labelsize="x-small")
        ax.tick_params(axis="both", which="minor", labelsize="xx-small")
        import matplotlib.colors

        palette = np.array(
            [
                (1, 1, 1, 0),
                (218 / 255, 215 / 255, 205 / 255, 1),
                (163 / 255, 177 / 255, 138 / 255, 1),
                (88 / 255, 129 / 255, 87 / 255, 1),
                (58 / 255, 9 / 255, 64 / 255, 1),
                (52 / 255, 78 / 255, 65 / 255, 1),
            ]
        )

        cmap = matplotlib.colors.ListedColormap(colors=palette)
        bounds = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

        im = ax.imshow(resized_confidence_map, cmap=cmap, norm=norm)
        cax = fig.add_axes(
            [
                ax.get_position().x1 + 0.01,
                ax.get_position().y0,
                0.02,
                ax.get_position().height,
            ]
        )

        cbar = plt.colorbar(
            im,
            cax=cax,
            extend="both",
            ticks=bounds,
            spacing="proportional",
            orientation="vertical",
        )
        cbar.set_label("Confidence", size="x-small")
        cbar.ax.tick_params(labelsize="xx-small")

        if output_path is not None:
            plt.savefig(
                os.path.join(output_path, "canopy_mask.jpg"), bbox_inches="tight"
            )

        if output_path is None:
            plt.show()

    def __str__(self) -> str:
        """String representation, returns canopy cover for image."""
        return (
            f"ProcessedSegmentationResult(n_trees={len(self.get_local_maxima())},"
            f" canopy_cover={self.canopy_cover:.4f})"
        )
