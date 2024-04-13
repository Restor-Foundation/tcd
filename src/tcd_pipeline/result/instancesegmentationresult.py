import logging
import math
import os
from typing import Any, Optional, Union

import cv2
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

from tcd_pipeline.postprocess.processedinstance import (
    ProcessedInstance,
    dump_instances_coco,
)
from tcd_pipeline.util import Vegetation

from .processedresult import ProcessedResult

logger = logging.getLogger(__name__)


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
        self.valid_mask = None
        self.prediction_time_s = -1
        self.config = config

        self.valid_window = rasterio.windows.from_bounds(
            *self.image.bounds, transform=self.image.transform
        )

        self.set_threshold(confidence_threshold)

    def _filter_roi(self):
        if self.valid_region is not None:
            self.instances = [
                instance
                for instance in self.instances
                if instance.transformed_polygon(self.image.transform).intersects(
                    self.valid_region
                )
            ]

            self.valid_window = rasterio.features.geometry_window(
                self.image, [self.valid_region]
            )

            self.valid_mask = rasterio.features.geometry_mask(
                [self.valid_region],
                out_shape=self.image.shape,
                transform=self.image.transform,
                invert=True,
            )[self.valid_window.toslices()]

            logger.info("Valid region and masks generated.")

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
        show_canopy=False,
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

        reshape_factor = 1
        if max_pixels is not None:
            reshape_factor = min(
                max_pixels[0] / self.valid_window.height,
                max_pixels[1] / self.valid_window.width,
            )
            reshape_factor = min(reshape_factor, 1)

        shape = (
            math.ceil(self.valid_window.height * reshape_factor),
            math.ceil(self.valid_window.width * reshape_factor),
        )

        vis_image = self.image.read(
            out_shape=(self.image.count, shape[0], shape[1]),
            resampling=Resampling.bilinear,
            masked=True,
            window=self.valid_window,
        ).transpose(1, 2, 0)

        if self.valid_mask is not None:
            if reshape_factor != 1:
                vis_image = vis_image * np.expand_dims(
                    resize(self.valid_mask, shape), -1
                )
            else:
                vis_image = vis_image * np.expand_dims(self.valid_mask, -1)

        resized_tree_mask = tree_mask
        resized_canopy_mask = canopy_mask

        if reshape_factor < 1:
            resized_tree_mask = resize(tree_mask, shape)
            resized_canopy_mask = resize(canopy_mask, shape)

        ax.imshow(vis_image)

        resized_canopy_mask = canopy_mask
        if reshape_factor < 1:
            resized_canopy_mask = resize(self.canopy_mask, shape)

        if show_canopy:
            canopy_mask_image = np.zeros(
                (*resized_canopy_mask.shape, 4), dtype=np.uint8
            )
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
                f"Unknown model config type {type(self.config.model.config)}"
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

        if "metadata" in reader.dataset:
            conf_thresh = reader.dataset["metadata"].get("threshold", 0)
            config = reader.dataset["metadata"].get("config", {})
            pred_time = reader.dataset["metadata"].get("prediction_time_s", -1)
        else:
            conf_thresh = 0.2
            config = {}
            pred_time = -1

        res = cls(
            image,
            instances,
            confidence_threshold=conf_thresh,
            config=config,
        )
        res.prediction_time_s = pred_time

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
                try:
                    from tcd_pipeline.util import paste_array

                    paste_array(
                        mask,
                        instance.local_mask,
                        offset=(instance.bbox.minx, instance.bbox.miny),
                    )
                except:
                    logger.warning(f"Failed to process instance")

        if self.valid_mask is not None:
            mask = mask[self.valid_window.toslices()] * self.valid_mask

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
                    "class": (
                        "tree" if instance.class_index == Vegetation.TREE else "canopy"
                    ),
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

    def _repr_html_(self):
        # Save the plot to a SVG buffer
        from io import BytesIO

        buf = BytesIO()
        plt.imshow(self.tree_mask)
        plt.savefig(buf, format="svg")
        plt.tight_layout()
        plt.close()
        buf.seek(0)
        return buf.getvalue().decode("utf-8")
