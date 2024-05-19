import json
import logging
import os
from typing import Any, Optional, Union

import affine
import numpy as np
import numpy.typing as npt
import rasterio
import scipy
import shapely
import shapely.geometry
import torch
import torchvision
from detectron2.structures import Instances
from pycocotools import mask as coco_mask
from rasterio.windows import Window
from shapely.affinity import affine_transform, translate
from shapely.geometry import box

from tcd_pipeline.util import inset_box, mask_to_polygon, polygon_to_mask

logger = logging.getLogger(__name__)


class ProcessedInstance:
    """Contains a processed instance that is detected by the model. Contains the score the algorithm gave, a polygon for the object,
    a bounding box and a local mask (a boolean mask of the size of the bounding box)

    If compression is enabled, then instance masks are automatically stored as memory-efficient objects. Currently two options are
    possible, either using the coco API ('coco') or scipy sparse arrays ('sparse').
    """

    def __init__(
        self,
        score: Union[float, list[float]],
        bbox: box,
        class_index: int,
        compress: Optional[str] = "sparse",
        global_polygon: Optional[shapely.geometry.MultiPolygon] = None,
        local_mask: Optional[npt.NDArray] = None,
        label: Optional[int] = None,
    ):
        """Initializes the instance

        Args:
            score (float): score given to the instance, or if a list, interpret as per-class scores
            bbox (box): the bounding box of the object
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
        score: Union[float, list[float]],
        bbox: box,
        class_index: int,
        compress: Optional[str] = "sparse",
        global_polygon: Optional[shapely.geometry.MultiPolygon] = None,
        local_mask: Optional[npt.NDArray] = None,
        label: Optional[int] = None,
    ):
        """Updates the instance

        Args:
            score (float): score given to the instance
            bbox (box): the bounding box of the object
            class_index (int): the class index of the object
            compress (optional, str): array compression method, defaults to coco
            global_polygon (MultiPolygon): a shapely MultiPolygon describing the segmented object in global image coordinates
            local_mask (array): local 2D binary mask for the instance
            label (optional, int): label associated with the processedInstance
        """

        score = np.array(score).reshape((-1, 1)).flatten()
        self.class_scores = None

        if len(score) > 1:
            self.score = float(max(score))
            self.class_scores = score
        else:
            self.score = float(score[0])

        self.bbox = bbox
        self.compress = compress
        self._local_mask = None
        self.class_index = class_index
        self.label = label

        # For most cases, we only need to store the mask:
        if local_mask is not None:
            self.local_mask = local_mask.astype(bool)

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
            return mask.astype(bool)
        elif self.compress == "coco":
            return coco_mask.decode(mask).astype(bool)
        elif self.compress == "sparse":
            return mask.toarray().astype(bool)
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

            minx, miny, maxx, maxy = self.bbox.bounds
            height = int(maxy - miny)
            width = int(maxx - minx)

            local_polygon = translate(self._polygon, xoff=-minx, yoff=-miny)
            self.local_mask = polygon_to_mask(local_polygon, shape=(height, width))

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

    def transformed_polygon(
        self, transform: affine.Affine
    ) -> shapely.geometry.MultiPolygon:
        """Transform polygon to world coordinates given an affine transform (typically
        obtained from a rasterio image's .transform property)

        Args:
            transform (affine.Affine): affine transform

        Returns:
            polygon (shapely.geometry.MultiPolygon): the polygon in world coordinates
        """
        # Re-order rasterio affine transform to shapely and map pixels -> world
        t = transform
        transform = [t.a, t.b, t.d, t.e, t.xoff, t.yoff]
        return affine_transform(self.polygon, transform)

    def _create_polygon(self, mask: npt.NDArray) -> None:
        """Internal function to generate polygon associated with mask"""
        polygon = mask_to_polygon(mask)
        # Positive offset into full image
        minx, miny, _, _ = self.bbox.bounds
        self._polygon = translate(polygon, xoff=minx, yoff=miny)

    def get_pixels(
        self, image: Union[rasterio.DatasetReader, npt.NDArray]
    ) -> npt.NDArray:
        """Gets the pixel values of the image at the location of the object

        Args:
            image (np.array[int]): image

        Returns:
            np.array[int]: pixel values at the location of the object
        """

        minx, miny, maxx, maxy = self.bbox.bounds
        height = int(maxy - miny)
        width = int(maxx - minx)

        if isinstance(image, rasterio.DatasetReader):
            window = Window(minx, miny, width, height)
            roi = image.read(window=window)
        elif isinstance(image, npt.NDArray):
            roi = image[miny:maxy, minx:maxx]

        return roi[..., self.local_mask]

    def get_image(self, image):
        """Gets the masked image at the location of the object
        Args:
                image (np.array(int)): image
        Returns:
                np.array(int): pixel values at the location of the object
        """
        return image[self.bbox.bounds] * np.repeat(
            np.expand_dims(self.local_mask, axis=-1), 3, axis=-1
        )

    @classmethod
    def from_coco_dict(
        cls,
        annotation: dict,
        image_shape: tuple[int] = None,
        global_mask: bool = False,
    ):
        """
        Instantiates an instance from a COCO dictionary.

        Args:
            annotation (dict): COCO formatted annotation dictionary
            image_shape (int): shape of the image
            global_mask (bool): specifies whether masks are stored in local or global coordinates.
                                This is overridden if the annotation file specifies that global
                                coords are used.

        """

        score = annotation.get("score", 1)

        # Override score if we have per-class predictions
        if "class_scores" in annotation:
            score = annotation["class_scores"]

        label = annotation.get("label")

        minx, miny, width, height = annotation["bbox"]

        if image_shape is not None:
            width = min(width, image_shape[1] - minx)
            height = min(height, image_shape[0] - miny)

        bbox = box(minx, miny, minx + width, miny + height)
        class_index = annotation["category_id"]

        if annotation["iscrowd"] == 1:
            # If 'counts' is not RLE encoded we need to convert it.
            if isinstance(annotation["segmentation"]["counts"], list):
                height, width = annotation["segmentation"]["size"]
                rle = coco_mask.frPyObjects(annotation["segmentation"], width, height)
                annotation["segmentation"] = rle

            local_mask = coco_mask.decode(annotation["segmentation"])

            if global_mask and annotation["global"] == 0:
                logger.warning(
                    "Requesting a global mask, but the annotation format is in local coordinates"
                )
            elif global_mask or annotation["global"] == 1:
                # If the mask is stored in global coordinates, then we expect the encoded mask
                # to be the shape of the source image. Here, extract/crop the local mask from
                # the global one.
                minx, miny, maxx, maxy = bbox.bounds
                local_mask = local_mask[miny:maxy, minx:maxx]

            polygon = None

        else:
            # Polygon annotations are always global
            coords = np.array(annotation["segmentation"]).reshape((-1, 2))
            polygon = shapely.geometry.Polygon(coords)
            minx, miny, maxx, maxy = bbox.bounds
            height = maxy - miny
            width = maxx - minx
            local_polygon = translate(polygon, xoff=-minx, yoff=-miny)
            local_mask = polygon_to_mask(local_polygon, shape=(height, width))
            polygon = shapely.geometry.MultiPolygon([polygon])

        return cls(
            score,
            bbox,
            class_index,
            global_polygon=polygon,
            local_mask=local_mask,
            label=label,
        )

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
            global_mask (bool): store masks in global coords (possibly CPU and nmemory intensive to compute)
            image_shape (tuple(int, int), optional): image shape, must be provided if global masks are used

        Returns:
            dict: COCO format dictionary
        """
        annotation = {}
        annotation["id"] = instance_id
        annotation["image_id"] = image_id
        annotation["category_id"] = int(self.class_index)

        # Store both predicted class score, and class score vector
        # as other software might not know how to deal with per
        # class predictions
        annotation["score"] = float(self.score)

        if self.class_scores is not None:
            annotation["class_scores"] = [float(score) for score in self.class_scores]

        minx, miny, maxx, maxy = self.bbox.bounds
        height = maxy - miny
        width = maxx - minx

        annotation["label"] = self.label
        annotation["bbox"] = [
            float(minx),
            float(miny),
            float(width),
            float(height),
        ]
        annotation["area"] = float(height * width)
        annotation["segmentation"] = {}

        # If the polygon has holes:
        if (
            isinstance(self.polygon, shapely.geometry.MultiPolygon)
            and len(self.polygon.geoms) > 0
        ):
            # For simplicity, always store as a RLE mask
            annotation["iscrowd"] = 1

            if global_mask:
                assert image_shape is not None

                annotation["global"] = 1
                coco_mask = np.zeros(image_shape, dtype=bool)
                coco_mask[
                    miny : miny + self.local_mask.shape[0],
                    minx : minx + self.local_mask.shape[1],
                ] = self.local_mask
            else:
                annotation["global"] = 0
                coco_mask = self.local_mask

            annotation["segmentation"] = self._mask_encode(coco_mask)
            if not isinstance(annotation["segmentation"]["counts"], str):
                annotation["segmentation"]["counts"] = annotation["segmentation"][
                    "counts"
                ].decode("utf-8")
        else:
            # Polygons are always stored in global image coords
            annotation["global"] = 1
            annotation["iscrowd"] = 0

            try:
                exterior_coords = self.polygon.exterior.coords
            except AttributeError:
                exterior_coords = [
                    list(poly.exterior.coords) for poly in self.polygon.geoms
                ]

            annotation["segmentation"] = [
                coord for xy in exterior_coords for coord in xy
            ]

        return annotation

    def __str__(self) -> str:
        return f"ProcessedInstance(score={self.score:.4f}, class={self.class_index}, {str(self.bbox)})"


def dump_instances_coco(
    output_path: str,
    instances: Optional[list[ProcessedInstance]] = [],
    image_path: Optional[str] = None,
    categories: Optional[dict] = None,
    metadata: Optional[dict] = None,
) -> dict:
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
        metadata (dict, optional): Arbitrary metadata to store in the file. Defaults to None.
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

    results["metadata"] = metadata

    annotations = []

    for idx, instance in enumerate(instances):
        annotation = instance.to_coco_dict(
            instance_id=idx,
            image_shape=image_shape,
        )
        annotations.append(annotation)

    results["annotations"] = annotations

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as fp:
        json.dump(results, fp, indent=1)

    logger.debug(f"Saved predictions for tile to {os.path.abspath(output_path)}")

    return results


def non_max_suppression(
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

        minx, miny, maxx, maxy = instance.bbox.bounds

        x1, x2 = float(minx), float(maxx)
        y1, y2 = float(miny), float(maxy)

        boxes.append([x1, y1, x2, y2])
        scores.append(instance.score)
        global_indices.append(idx)

    if len(boxes) > 0:
        global_indices = np.array(global_indices)
        boxes = np.array(boxes, dtype=np.float32)

        scores = torch.Tensor(scores)
        boxes = torch.from_numpy(boxes)

        keep_indices = torchvision.ops.nms(boxes, scores, iou_threshold)

        return np.array([global_indices[keep_indices]]).flatten()

    else:
        return []
