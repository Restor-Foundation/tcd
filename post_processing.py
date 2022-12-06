import json
import logging
import os
import pickle
import shutil
import time
from collections import OrderedDict
from glob import glob
from typing import Any, Optional

import fiona
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pycocotools.coco
import rasterio
import scipy
import seaborn as sns
import shapely
import torch
import torchvision
from detectron2.structures import Instances
from genericpath import exists
from natsort import natsorted
from PIL import Image
from pycocotools import mask as coco_mask
from rasterio import features
from rasterio.enums import Resampling
from rasterio.windows import Window
from shapely.affinity import translate
from skimage.transform import resize
from torchgeo.datasets import BoundingBox
from tqdm.auto import tqdm

from util import Vegetation

logger = logging.getLogger(__name__)


def mask_to_polygon(mask: npt.NDArray[np.bool]) -> shapely.geometry.MultiPolygon:
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
        np.array(bool): Boolean mask of the segmented object
    """

    return features.rasterize([polygon], out_shape=shape)


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

    def window(self) -> Window:
        return Window(self.minx, self.miny, self.width, self.height)

    def __str__(self) -> str:
        return f"Bbox(minx={self.minx:.4f}, miny={self.miny:.4f}, maxx={self.maxx:.4f}, maxy={self.maxy:.4f})"


class ProcessedInstance:
    """Contains a processed instance that is detected by the model. Contains the score the algorithm gave, a polygon for the object,
    a bounding box and a local mask (a boolean mask of the size of the bounding box)

    If compression is enabled, then instance masks are automatically stored as memory-efficient objects. Currently two options are
    possible, either using the coco API ('coco') or scipy sparse arrays ('sparse').
    """

    def __init__(
        self,
        score: float,
        bbox: Bbox,
        class_index: int,
        compress: Optional[str] = "sparse",
        global_polygon: Optional[shapely.geometry.MultiPolygon] = None,
        local_mask: Optional[npt.NDArray] = None,
        label: Optional[int] = None,
    ):
        """Initializes the instance

        Args:
            score (float): score given to the instance
            bbox (Bbox): the bounding box of the object
            class_index (int): the class index of the object
            compress (optional, str): array compression method, defaults to coco
            global_polygon (MultiPolygon): a shapely MultiPolygon describing the segmented object in global image coordinates
            local_mask (array): local 2D binary mask for the instance
            label (optional, int): label associated with the processedInstance
        """
        self.update(
            score, bbox, class_index, compress, global_polygon, local_mask, label
        )

    def update(
        self,
        score: float,
        bbox: Bbox,
        class_index: int,
        compress: Optional[str] = "sparse",
        global_polygon: Optional[shapely.geometry.MultiPolygon] = None,
        local_mask: Optional[npt.NDArray] = None,
        label: Optional[int] = None,
    ):
        """Updates the instance

        Args:
            score (float): score given to the instance
            bbox (Bbox): the bounding box of the object
            class_index (int): the class index of the object
            compress (optional, str): array compression method, defaults to coco
            global_polygon (MultiPolygon): a shapely MultiPolygon describing the segmented object in global image coordinates
            local_mask (array): local 2D binary mask for the instance
            label (optional, int): label associated with the processedInstance
        """
        self.score = float(score)
        self.bbox = bbox
        self.compress = compress
        self._local_mask = None
        self.class_index = class_index
        self.label = label

        # For most cases, we only need to store the mask:
        if local_mask is not None:
            self.local_mask = local_mask.astype(np.bool)

        # If a polygon is supplied store it, but default None
        self._polygon = global_polygon

    def _compress(self, mask: npt.NDArray) -> Any:
        """Internal method to compress a local annotation mask
        use 'coco' to store RLE encoded masks. Use 'sparse'
        to store scipy sparse arrays, or None to disable.

        Args:
            mask (array): mask array

        Returns:
            Any: compressed mask

        """
        if self.compress is None:
            return mask
        elif self.compress == "coco":
            return coco_mask.encode(np.asfortranarray(mask))
        elif self.compress == "sparse":
            return scipy.sparse.csr_matrix(mask)
        else:
            raise NotImplementedError(
                f"{self.compress} is not a valid compression method"
            )

    def _decompress(self, mask: Any) -> npt.NDArray:
        """Internal method to decompress a local annotation mask

        Args:
            mask (Any): compressed mask

        Returns:
            np.array: uncompressed mask
        """

        if mask is None:
            return mask

        if self.compress is None:
            return mask.astype(np.bool)
        elif self.compress == "coco":
            return coco_mask.decode(mask).astype(np.bool)
        elif self.compress == "sparse":
            return mask.toarray().astype(np.bool)
        else:
            raise NotImplementedError(
                f"{self.compress} is not a valid compression method"
            )

    @property
    def local_mask(self) -> npt.NDArray:
        """Returns the local annotation mask.

        Returns:
            np.array: local annotation mask
        """
        if self._local_mask is None:

            assert self._polygon is not None

            local_polygon = translate(
                self._polygon, xoff=-self.bbox.minx, yoff=-self.bbox.miny
            )
            self.local_mask = polygon_to_mask(
                local_polygon, shape=(self.bbox.height, self.bbox.width)
            )

        return self._decompress(self._local_mask)

    @local_mask.setter
    def local_mask(self, local_mask):
        """Internal function for setting local annotation mask, compresses
        using the specified method (e.g. coco, pickle)
        """
        self._local_mask = self._compress(local_mask)

    @property
    def polygon(self) -> shapely.geometry.MultiPolygon:
        """Returns the polygon associated with this instance, creates it if
        it doesn't exist.
        """
        if self._polygon is None:

            assert self.local_mask is not None
            self._create_polygon(self.local_mask)

        return self._polygon

    def _create_polygon(self, mask: npt.NDArray) -> None:
        """Internal function to generate polygon associated with mask"""
        polygon = mask_to_polygon(mask)
        # Positive offset into full image
        self._polygon = translate(polygon, xoff=self.bbox.minx, yoff=self.bbox.miny)

    def get_pixels(self, image: npt.NDArray) -> npt.NDArray:
        """Gets the pixel values of the image at the location of the object

        Args:
            image (np.array[int]): image

        Returns:
            np.array[int]: pixel values at the location of the object
        """
        return image[self.bbox.miny : self.bbox.maxy, self.bbox.minx : self.bbox.maxx][
            self.local_mask
        ]

    def get_image(self, image):
        """Gets the masked image at the location of the object
        Args:
                image (np.array(int)): image
        Returns:
                np.array(int): pixel values at the location of the object
        """
        return image[
            self.bbox.miny : self.bbox.maxy, self.bbox.minx : self.bbox.maxx
        ] * np.repeat(np.expand_dims(self.local_mask, axis=-1), 3, axis=-1)

    @classmethod
    def from_coco_dict(
        self, annotation: dict, image_shape: list[int] = None, global_mask: bool = False
    ):
        """
        Instantiates an instance from a COCO dictionary.

        Args:
            annotation (dict): COCO formatted annotation dictionary
            image_shape (int): shape of the image
            global_mask (bool): specifies whether masks are stored in local or global coordinates

        """

        score = annotation.get("score", 1)
        label = annotation.get("label")

        minx, miny, width, height = annotation["bbox"]

        if image_shape is not None:
            width = min(width, image_shape[1] - minx)
            height = min(height, image_shape[0] - miny)

        bbox = Bbox(minx, miny, minx + width, miny + height)
        class_index = annotation["category_id"]

        if annotation["iscrowd"] == 1:
            # If 'counts' is not RLE encoded we need to convert it.
            if isinstance(annotation["segmentation"]["counts"], list):
                height, width = annotation["segmentation"]["size"]
                rle = coco_mask.frPyObjects(annotation["segmentation"], width, height)
                annotation["segmentation"] = rle

            local_mask = coco_mask.decode(annotation["segmentation"])
            if global_mask:
                local_mask = local_mask[bbox.miny : bbox.maxy, bbox.minx : bbox.maxx]
        else:
            coords = np.array(annotation["segmentation"]).reshape((-1, 2))
            polygon = shapely.geometry.Polygon(coords)
            local_polygon = translate(polygon, xoff=-bbox.minx, yoff=-bbox.miny)
            local_mask = polygon_to_mask(local_polygon, shape=(bbox.height, bbox.width))
            self._polygon = shapely.geometry.MultiPolygon([polygon])

        return self(score, bbox, class_index, local_mask=local_mask, label=label)

    def _mask_encode(self, mask: npt.NDArray) -> dict:
        """
        Internal function to encode an annotation mask in COCO format. Currently
        this uses pycocotools, but faster implementations may be available in the
        future.

        Args:
            annotation (npt.NDArray): 2D annotation mask

        Returns:
            dict: encoded segmentation object

        """
        return coco_mask.encode(np.asfortranarray(mask))

    def to_coco_dict(
        self,
        image_id: int = 0,
        instance_id: int = 0,
        global_mask: bool = False,
        image_shape: Optional[tuple[int, int]] = None,
    ) -> dict:
        """Outputs a COCO dictionary in global image coordinates. Will automatically
        pick whether to store a polygon (if the annotation is simple) or a RLE
        encoded mask. You can store masks in local or global coordinates.

        Args:
            image_id (int): image ID that this annotation corresponds to
            instance_id (int): instance ID - should be unique
            global_mask (bool): store masks in global coords (CPU intensive to compute)
            image_shape (tuple(int, int), optional): image shape, must be provided if global masks are used

        Returns:
            dict: COCO format dictionary
        """
        annotation = {}
        annotation["id"] = instance_id
        annotation["image_id"] = image_id
        annotation["category_id"] = int(self.class_index)
        annotation["score"] = float(self.score)
        annotation["label"] = self.label
        annotation["bbox"] = [
            float(self.bbox.minx),
            float(self.bbox.miny),
            float(self.bbox.width),
            float(self.bbox.height),
        ]
        annotation["area"] = float(self.bbox.area)
        annotation["segmentation"] = {}

        # For simplicity, always store as a RLE mask
        annotation["iscrowd"] = 1

        if global_mask:
            assert image_shape is not None

            annotation["global"] = 1
            coco_mask = np.zeros(image_shape, dtype=bool)
            coco_mask[
                self.bbox.miny : self.bbox.miny + self.local_mask.shape[0],
                self.bbox.minx : self.bbox.minx + self.local_mask.shape[1],
            ] = self.local_mask
        else:
            annotation["global"] = 0
            coco_mask = self.local_mask

        annotation["segmentation"] = self._mask_encode(coco_mask)
        if not isinstance(annotation["segmentation"]["counts"], str):
            annotation["segmentation"]["counts"] = annotation["segmentation"][
                "counts"
            ].decode("utf-8")

        return annotation

    def __str__(self) -> str:
        return f"ProcessedInstance(score={self.score:.4f}, class={self.class_index}, {str(self.bbox)})"


def dump_instances_coco(
    output_path: str,
    instances: Optional[list[ProcessedInstance]] = [],
    image_path: Optional[str] = None,
    categories: Optional[dict] = None,
    threshold: Optional[float] = 0,
) -> None:
    """Store a list of instances as a COCO formatted JSON file.

    If an image path is provided then some info will be stored in the file. This utility
    is designed to aid with serialising tiled predictions. Typically COCO
    format results just reference an image ID, however for predictions over
    large orthomosaics we typically only have a single image, so the ID is
    set here to zero and we provide information in the annotation file
    directly. This is just for compatibility.

    Args:
        output_path (str): Path to output json file. Intermediate folders
        will be created if necessary.
        instances (list[ProcessedInstance]): List of instances to store.
        image_path (str, optional): Path to image. Defaults to None.
        categories (dict of int: str, optional): Class map from ID to name. Defaults to None
        threshold (float, optional): Confidence threshold to store
    """

    results = {}
    image_shape = None

    if image_path is not None:

        image_dict = {}
        image_dict["id"] = 0
        image_dict["file_name"] = os.path.basename(image_path)

        with rasterio.open(image_path, "r+") as src:
            image_dict["width"] = src.width
            image_dict["height"] = src.height
            image_shape = src.shape

        results["images"] = [image_dict]

    if categories is not None:
        out_categories = []

        for key in categories:
            category = {}
            category["id"] = key
            category["name"] = categories[key]
            category["supercategory"] = categories[key]
            out_categories.append(category)

        results["categories"] = out_categories

    if threshold is not None:
        results["threshold"] = threshold

    annotations = []

    for idx, instance in enumerate(instances):
        annotation = instance.to_coco_dict(instance_id=idx, image_shape=image_shape)
        annotations.append(annotation)

    results["annotations"] = annotations

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as fp:
        json.dump(results, fp, indent=1)

    logger.debug(f"Saved predictions for tile to {os.path.abspath(output_path)}")


class ProcessedResult:
    """A processed result of a model. It contains all trees separately and also a global tree mask, canopy mask and image"""

    def __init__(
        self,
        image: npt.NDArray,
        instances: Optional[list] = [],
        confidence_threshold: int = 0,
    ) -> None:
        """Initializes the Processed Result

        Args:
            image (np.array[int]): source image that instances are referenced to
            instances (List[ProcessedInstance], optional): list of all instances. Defaults to []].
            confidence_threshold (int): confidence threshold for retrieving instances
        """
        self.image = image
        self.instances = instances
        self.set_threshold(confidence_threshold)

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
        color_trees: Optional[tuple[int, int, int]] = (204, 0, 0),
        color_canopy: Optional[tuple[int, int, int]] = (0, 0, 204),
        alpha: Optional[float] = 0.3,
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

        reshape_factor = 1
        if max_pixels is not None:
            reshape_factor = min(
                max_pixels[0] / self.image.height, max_pixels[1] / self.image.width
            )
            reshape_factor = min(reshape_factor, 1)

        shape = (
            int(self.image.height * reshape_factor),
            int(self.image.width * reshape_factor),
        )

        self.vis_image = self.image.read(
            out_shape=(self.image.count, shape[0], shape[1]),
            resampling=Resampling.bilinear,
        ).transpose(1, 2, 0)

        ax.imshow(self.vis_image)

        canopy_mask_image = np.zeros((shape[0], shape[1], 4), dtype=np.uint8)
        resized_canopy_mask = self.canopy_mask
        if reshape_factor < 1:
            resized_canopy_mask = resize(self.canopy_mask, shape)

        canopy_mask_image[resized_canopy_mask > 0] = list(color_canopy) + [255]
        ax.imshow(canopy_mask_image, alpha=alpha)

        tree_mask_image = np.zeros((shape[0], shape[1], 4), dtype=np.uint8)

        resized_tree_mask = self.tree_mask
        if reshape_factor < 1:
            resized_tree_mask = resize(self.tree_mask, shape)

        tree_mask_image[resized_tree_mask > 0] = list(color_trees) + [255]
        ax.imshow(tree_mask_image, alpha=alpha)

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
        image_path: Optional[str] = None,
        file_name: Optional[str] = "results.json",
    ) -> None:
        """Serialise results to a COCO JSON file.

        Args:
            output_folder (str): output folder
            overwrite (bool, optional): overwrite existing data, defaults True
            image_path (str): path to source image, default None
            file_name (str, optional): file name, defaults to results.json
        """

        logger.info(f"Serialising results to {output_folder}/{file_name}")
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, file_name)

        if os.path.exists(output_path) and not overwrite:
            logger.error(
                f"Output file already exists {output_path}, will not overwrite."
            )

        categories = {
            Vegetation.TREE: Vegetation.TREE.name.lower(),
            Vegetation.CANOPY: Vegetation.CANOPY.name.lower(),
        }

        dump_instances_coco(
            output_path,
            instances=self.instances,
            image_path=image_path,
            categories=categories,
            threshold=self.confidence_threshold,
        )

    @classmethod
    def load_serialisation(
        self,
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

        threshold = reader.dataset.get("threshold", 0)

        return self(image, instances, threshold)

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
        output_folder: str,
        image_path: Optional[str] = None,
        suffix: Optional[str] = "",
    ) -> None:
        """Save prediction masks for tree and canopy. If a source image is provided
        then it is used for georeferencing the output masks.

        Args:
            output_folder (str): folder to store data
            image_path (str, optional): source image
            suffix (str, optional): mask filename suffix

        """

        if image_path is not None:
            with rasterio.open(image_path, "r+") as src:
                out_meta = src.meta
                out_transform = src.transform
                out_height = src.height
                out_width = src.width
                out_crs = src.crs
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

            with rasterio.open(
                os.path.join(output_folder, f"tree_mask{suffix}.tif"),
                "w",
                nbits=1,
                **out_meta,
            ) as dest:
                dest.write(self.tree_mask, indexes=1)

            with rasterio.open(
                os.path.join(output_folder, f"canopy_mask{suffix}.tif"),
                "w",
                nbits=1,
                **out_meta,
            ) as dest:
                dest.write(self.canopy_mask, indexes=1)

        else:
            logger.warning(
                "No base image provided, output masks will not be georeferenced"
            )
            tree_mask = Image.fromarray(self.tree_mask)
            tree_mask.save(
                os.path.join(output_folder, "tree_mask.tif"), compress="packbits"
            )

            canopy_mask = Image.fromarray(self.canopy_mask)
            canopy_mask.save(
                os.path.join(output_folder, "canopy_mask.tif"), compress="packbits"
            )

    def set_threshold(self, new_threshold: int) -> None:
        """Sets the threshold of the ProcessedResult, also regenerates
        prediction masks

        Args:
            new_threshold (double): new confidence threshold
        """
        self.confidence_threshold = new_threshold
        self.canopy_mask = self._generate_mask(Vegetation.CANOPY)
        self.tree_mask = self._generate_mask(Vegetation.TREE)

    def save_shapefile(
        self, out_path: str, image_path: str, indices: Vegetation = None
    ) -> None:
        """Save instances to a georeferenced shapefile.

        Args:
            out_path (str): output file path
            image_path (str): path to georeferenced image
            class_index (Vegetation, optional): on
        """

        schema = {
            "geometry": "MultiPolygon",
            "properties": {"score": "float", "class": "str"},
        }

        src = rasterio.open(image_path, "r")
        with fiona.open(
            out_path, "w", "ESRI Shapefile", schema=schema, crs=src.crs.wkt
        ) as layer:
            for instance in self.get_instances():

                elem = {}

                # Re-order rasterio affine transform to shapely and map pixels -> world
                t = src.transform
                transform = [t.a, t.b, t.d, t.e, t.xoff, t.yoff]
                geo_poly = shapely.affinity.affine_transform(
                    instance.polygon, transform
                )

                elem["geometry"] = shapely.geometry.mapping(geo_poly)
                elem["properties"] = {
                    "score": instance.score,
                    "class": "tree"
                    if instance.class_index == Vegetation.TREE
                    else "canopy",
                }

                layer.write(elem)

    def __str__(self) -> str:
        """String representation, returns canopy and tree cover for image."""
        canopy_cover = np.count_nonzero(self.canopy_mask) / np.prod(
            self.image.shape[:2]
        )
        tree_cover = np.count_nonzero(self.tree_mask) / np.prod(self.image.shape[:2])
        return f"ProcessedResult(n_trees={len(self.get_trees())}, canopy_cover={canopy_cover:.4f}, tree_cover={tree_cover:.4f})"


class PostProcessor:
    """Processes the result of the modelRunner"""

    def __init__(self, config: dict, image: Optional[rasterio.DatasetReader] = None):
        """Initializes the PostProcessor

        Args:
            config (DotMap): the configuration
            image (DatasetReader): input rasterio image
        """
        self.config = config
        self.threshold = config.postprocess.confidence_threshold

        self.cache_root = config.postprocess.cache_folder
        self.cache_folder = None

        if image is not None:
            self.initialise(image)

    def initialise(self, image) -> None:
        """Initialise the processor for a new image and creates cache
        folders if required.

        Args:
            image (DatasetReader): input rasterio image
        """
        self.untiled_instances = []
        self.image = image
        self.tile_count = 0
        self.tiled_bboxes = []
        self.cache_folder = os.path.join(
            self.cache_root,
            os.path.splitext(os.path.basename(self.image.name))[0] + "_cache",
        )

        if self.config.postprocess.stateful:

            if os.path.exists(self.cache_folder):
                logger.warning("Cache folder exists already")
                self.clear_cache()

            os.makedirs(self.cache_folder, exist_ok=True)
            logger.info(f"Caching to {self.cache_folder}")

    def clear_cache(self):
        """Clear cache. Warning: there are no checks here, the set cache folder and its
        contents will be deleted.
        """
        if self.config.postprocess.stateful:
            logger.warning("Clearing cache folder")
            shutil.rmtree(self.cache_folder)
        else:
            logger.warning("Processor is not in stateful mode.")

    def _get_proper_bbox(self, bbox: Optional[BoundingBox] = None):
        """Returns a pixel-coordinate bbox of an image given a Torchgeo bounding box.

        Args:
            bbox (BoundingBox): Bounding box from torchgeo query. Defaults to None (bbox is entire image)

        Returns:
            Bbox: Bbox with correct orientation compared to the image
        """
        if bbox is None:
            minx, miny = 0, 0
            maxx, maxy = self.image.shape[0], self.image.shape[1]
        else:
            miny, minx = self.image.index(bbox.minx, bbox.miny)
            maxy, maxx = self.image.index(bbox.maxx, bbox.maxy)
        # Sort coordinates if necessary
        if miny > maxy:
            miny, maxy = maxy, miny

        if minx > maxx:
            minx, maxx = maxx, minx

        return Bbox(minx=minx, miny=miny, maxx=maxx, maxy=maxy)

    def process_untiled_result(self, result: Instances) -> ProcessedResult:
        """Processes results outputted by Detectron without tiles

        Args:
            results (Instances): Results predicted by the detectron model

        Returns:
            ProcessedResult: ProcessedResult of the segmentation task
        """
        return self.process_tiled_result([[result, None]])

    def detectron_to_instances(
        self,
        result: tuple[Instances, BoundingBox],
        edge_tolerance: Optional[int] = 5,
    ) -> list[ProcessedInstance]:
        """Convert a Detectron2 result to a list of ProcessedInstances

        Args:
            result (tuple[Instance, Bbox]): result containing the Detectron instances and the bounding box
            of the tile
            edge_tolerance (int): threshold to remove trees at the edge of the image, not applied to canopy

        Returns
            list[ProcessedInstance]: list of instances

        """

        tstart = time.time()

        instances, bbox = result
        proper_bbox = self._get_proper_bbox(bbox)
        out = []

        for instance_index in range(len(instances)):
            class_idx = int(instances.pred_classes[instance_index])

            bbox_instance_tiled = (
                instances.pred_boxes[instance_index].tensor[0].int().cpu().numpy()
            )

            global_mask = instances.pred_masks[instance_index].cpu().numpy()
            pred_height, pred_width = global_mask.shape

            bbox_instance = Bbox(
                minx=proper_bbox.minx + bbox_instance_tiled[0],
                miny=proper_bbox.miny + bbox_instance_tiled[1],
                maxx=proper_bbox.minx + bbox_instance_tiled[2],
                maxy=proper_bbox.miny + bbox_instance_tiled[3],
            )

            local_mask = global_mask[
                bbox_instance_tiled[1] : bbox_instance_tiled[3],
                bbox_instance_tiled[0] : bbox_instance_tiled[2],
            ]

            # Filter tree boxes that touch the edge of the tile
            if class_idx == Vegetation.TREE:
                if bbox_instance_tiled[0] < edge_tolerance:
                    continue
                if bbox_instance_tiled[1] < edge_tolerance:
                    continue
                if bbox_instance_tiled[2] > (pred_height - edge_tolerance):
                    continue
                if bbox_instance_tiled[3] > (pred_width - edge_tolerance):
                    continue

            new_instance = ProcessedInstance(
                class_index=class_idx,
                local_mask=local_mask,
                bbox=bbox_instance,
                score=instances.scores[instance_index],
                compress="sparse",
            )

            out.append(new_instance)

        telapsed = time.time() - tstart
        logger.debug(f"Processed instances {len(instances)} in {telapsed:1.2f}s")

        return out

    def cache_tiled_result(self, result: tuple[Instances, BoundingBox]) -> None:

        """Cache a single tile result

        Args:
            result (tuple[Instance, Bbox]): result containing the Detectron instances and the bounding box
            of the tile

        """

        processed_instances = self.detectron_to_instances(result)

        categories = {
            Vegetation.TREE: Vegetation.TREE.name.lower(),
            Vegetation.CANOPY: Vegetation.CANOPY.name.lower(),
        }

        self.tile_count += 1
        self.tiled_bboxes.append(self._get_proper_bbox(result[1]))
        cache_format = self.config.postprocess.cache_format

        if cache_format == "coco":
            dump_instances_coco(
                os.path.join(self.cache_folder, f"{self.tile_count}_instances.json"),
                instances=processed_instances,
                image_path=self.image.name,
                categories=categories,
            )
        elif cache_format == "pickle":
            with open(
                os.path.join(self.cache_folder, f"{self.tile_count}_instances.pkl"),
                "wb",
            ) as fp:
                pickle.dump(processed_instances, fp)
        else:
            raise NotImplementedError(f"Cache format {cache_format} is unsupported")

        if self.config.postprocess.debug_images:
            proper_bbox = self._get_proper_bbox(result[1])

            kwargs = self.image.meta.copy()
            window = proper_bbox.window()

            kwargs.update(
                {
                    "height": window.height,
                    "width": window.width,
                    "transform": rasterio.windows.transform(
                        window, self.image.transform
                    ),
                    "compress": "jpeg",
                }
            )

            with rasterio.open(
                os.path.join(self.cache_folder, f"{self.tile_count}_tile.tif"),
                "w",
                **kwargs,
            ) as dst:
                dst.write(self.image.read(window=window))

    def _load_cache_coco(self, cache_file: str) -> list[ProcessedInstance]:
        """Load cached results from COCO format

        Args:
            cache_file (str): Cache filename

        Returns:
            list[ProcessedInstance]: instances

        """

        out = []

        with open(cache_file, "r") as fp:
            annotations = json.load(fp)

            if annotations.get("image"):
                if annotations["image"]["file_name"] != os.path.basename(
                    self.image.name
                ):
                    logger.warning(
                        "No image information in file, skipping out of caution."
                    )

            for annotation in annotations["annotations"]:
                instance = ProcessedInstance.from_coco_dict(
                    annotation, self.image.shape
                )
                out.append(instance)

        return out

    def _load_cache_pickle(self, cache_file: str) -> list[ProcessedInstance]:
        """Load cached results that were pickled.

        Args:
            cache_file (str): Cache filename

        Returns:
            list[ProcessedInstance]: instances

        """

        with open(cache_file, "rb") as fp:
            annotations = pickle.load(fp)

        return annotations

    def merge_instance(
        self,
        new_instance: ProcessedInstance,
        tile_instance: int,
        iou_threshold: Optional[int] = 0.2,
    ):
        """Merges instances from other tiles if they overlap

        Args:
            new_instance (ProcessedInstanace): Instance from the tile that is currently added
            tile_instance (int): Tile from which the instance is
            iou_threshold (Optional[int], optional): Threshold for merging. Defaults to 0.2.

        Returns:
            ProcessedInstance: The instance that needs to be added to the list
        """
        for tile_index, tile in enumerate(self.tiled_bboxes[:tile_instance]):
            if tile.overlap(self.tiled_bboxes[tile_instance]):
                for i, other_instance in enumerate(self.untiled_instances[tile_index]):
                    if (
                        other_instance.class_index == new_instance.class_index
                        and other_instance.bbox.overlap(new_instance.bbox)
                    ):
                        intersection = other_instance.polygon.intersection(
                            new_instance.polygon
                        )
                        union = other_instance.polygon.union(new_instance.polygon)
                        iou = intersection.area / union.area
                        if iou > iou_threshold:
                            new_bbox = Bbox(
                                minx=min(
                                    other_instance.bbox.minx, new_instance.bbox.minx
                                ),
                                miny=min(
                                    other_instance.bbox.miny, new_instance.bbox.miny
                                ),
                                maxx=max(
                                    other_instance.bbox.maxx, new_instance.bbox.maxx
                                ),
                                maxy=max(
                                    other_instance.bbox.maxy, new_instance.bbox.maxy
                                ),
                            )
                            new_score = (
                                new_instance.score * new_instance.polygon.area
                                + other_instance.score * other_instance.polygon.area
                            )
                            new_score /= (
                                new_instance.polygon.area + other_instance.polygon.area
                            )

                            self.untiled_instances[tile_index][i].update(
                                score=new_score,
                                bbox=new_bbox,
                                class_index=new_instance.class_index,
                                compress=new_instance.compress,
                                global_polygon=union,
                                label=other_instance.label,
                            )

                            return self.untiled_instances[tile_index][i]
        return new_instance

    def merge_instances_other_tiles(
        self, new_instances: list[ProcessedInstance], tile_instances: int
    ):
        """Merges the instances from a new tile into the predictions

        Args:
            new_instances (list[ProcessedInstance]): List of processed instances
            tile_instances (int): Tile from which the instances are
        """
        new_annotations = []
        for annotation in new_instances:
            if (
                Vegetation(annotation.class_index).name.lower()
                in self.config.postprocess.merge_classes
            ):
                merged = self.merge_instance(
                    annotation, tile_instances, self.config.postprocess.iou_tiles
                )
                new_annotations.append(merged)
            else:
                new_annotations.append(annotation)

        self.untiled_instances.append(new_annotations)

    def process_cached(self) -> None:
        """Load cached predictions. Should be called once the image has been predicted
        and prior to tile processing.

        This function will look in the preset cache_folder
        for files with the correct format. It does not do any kind of check to see if
        these results are consistent with one another, so it is up to you to make sure
        that you cache folder is clean. If you don't, you might mix up detections from
        multiple predictions / tiles sizes.

        """

        logger.info(
            f"Looking for cached files in: {os.path.abspath(self.cache_folder)}"
        )

        cache_format = self.config.postprocess.cache_format

        if cache_format == "pickle":
            cache_glob = f"*_instances.pkl"
        elif cache_format == "coco":
            cache_glob = f"*_instances.json"
        else:
            raise NotImplementedError(f"Cache format {cache_format} is unsupported")

        cache_files = natsorted(glob(os.path.join(self.cache_folder, cache_glob)))
        self.untiled_instances = []

        for i in tqdm(range(len(cache_files))):
            cache_file = cache_files[i]  # no enumerate because uglier tqdm
            if cache_format == "coco":
                annotations = self._load_cache_coco(cache_file)
            elif cache_format == "pickle":
                annotations = self._load_cache_pickle(cache_file)

            self.merge_instances_other_tiles(annotations, i)

            logger.debug(f"Loaded {len(annotations)} instances from {cache_file}")

    def append_tiled_result(self, result: tuple[Instances, BoundingBox]) -> None:
        """
        Adds a detectron2 result to the processor

        Args:
            result (Any): detectron result
        """
        annotations = self.detectron_to_instances(result)
        self.tiled_bboxes.append(self._get_proper_bbox(result[1]))
        self.tile_count += 1
        self.merge_instances_other_tiles(annotations, self.tile_count)

    def _collect_tiled_result(self, results: tuple[Instances, BoundingBox]) -> None:
        """Collects all segmented objects that are predicted and puts them in a ProcessedResult. Also creates global masks for trees and canopies

        Args:
            results (List[[Instances, BoundingBox]]): Results predicted by the detectron model
            threshold (float, optional): threshold for adding the detected objects. Defaults to 0.5.

        """

        for result in results:
            self.append_tiled_result(result)

    def non_max_suppression(
        self,
        instances: list[ProcessedInstance],
        class_index: int,
        iou_threshold: float = 0.8,
    ) -> list[int]:
        """Perform non-maximum suppression on the list of input instances

        Args:
            instances (list(ProcessedInstance)): instances to filter
            class_index (int): class of interest
            iou_threshold (float, optional): IOU threshold Defaults to 0.8.

        Returns:
            list[int]: List of indices of boxes to keep
        """

        boxes = []
        scores = []
        global_indices = []

        for idx, instance in enumerate(instances):

            if instance.class_index != class_index:
                continue

            x1, x2 = float(instance.bbox.minx), float(instance.bbox.maxx)
            y1, y2 = float(instance.bbox.miny), float(instance.bbox.maxy)

            boxes.append([x1, y1, x2, y2])
            scores.append(instance.score)
            global_indices.append(idx)

        if len(boxes) > 0:

            global_indices = np.array(global_indices)
            boxes = np.array(boxes, dtype=np.float32)

            scores = torch.Tensor(scores)
            boxes = torch.from_numpy(boxes)

            keep_indices = torchvision.ops.nms(boxes, scores, iou_threshold)
            return global_indices[keep_indices]

        else:
            return []

    def process_tiled_result(
        self, results: list[tuple[Instances, BoundingBox]] = None
    ) -> ProcessedResult:
        """Processes the result of the detectron model when the tiled version was used

        Args:
            results (List[[Instances, BoundingBox]]): Results predicted by the detectron model. Defaults to None.

        Returns:
            ProcessedResult: ProcessedResult of the segmentation task
        """

        logger.info("Collecting results")

        assert self.image is not None

        if results is not None:
            self._collect_tiled_result(results)

        self.all_instances = []
        for tile in range(len(self.untiled_instances)):
            for instance in self.untiled_instances[tile]:
                if (
                    instance not in self.all_instances
                ):  # Merged instances are in multiple tiles
                    self.all_instances.append(instance)

        self.merged_instances = []

        if self.config.postprocess.use_nms:
            logger.info("Running non-max suppression")

            nms_indices = self.non_max_suppression(
                self.all_instances,
                class_index=Vegetation.TREE,
                iou_threshold=self.config.postprocess.iou_threshold,
            )

            for idx in nms_indices:
                self.merged_instances.append(self.all_instances[idx])

            nms_indices = self.non_max_suppression(
                self.all_instances,
                class_index=Vegetation.CANOPY,
                iou_threshold=self.config.postprocess.iou_threshold,
            )

            for idx in nms_indices:
                self.merged_instances.append(self.all_instances[idx])
        else:
            self.merged_instances = self.all_instances

        logger.info("Result collection complete")

        return ProcessedResult(image=self.image, instances=self.merged_instances)
