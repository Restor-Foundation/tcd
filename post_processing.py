import json
import logging
import os
import time
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import scipy
import shapely
import torch
import torchvision
from genericpath import exists
from PIL import Image
from pycocotools import mask as coco_mask
from rasterio import features
from shapely.affinity import translate
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def dump_instances_coco(output_path, instances, image_path=None, categories=None):
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
    """

    results = {}

    if image_path is not None:

        image_dict = {}
        image_dict["id"] = 0
        image_dict["file_name"] = os.path.basename(image_path)

        with rasterio.open(image_path, "r+") as src:
            image_dict["width"] = src.width
            image_dict["height"] = src.height

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

    annotations = []
    for idx, instance in tqdm(enumerate(instances)):

        annotation = instance.to_coco_dict(
            image_shape=[src.height, src.width], instance_id=idx
        )
        annotations.append(annotation)

    results["annotations"] = annotations

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as fp:
        json.dump(results, fp, indent=1)


def mask_to_polygon(mask):
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


def polygon_to_mask(polygon, shape):
    """Converts the mask of an object to a MultiPolygon

    Args:
        mask (np.array(bool)): Boolean mask of the segmented object

    Returns:
        MultiPolygon: Shapely MultiPolygon describing the object
    """

    return features.rasterize([polygon], out_shape=shape)


class Bbox:
    """A bounding box with integer coordinates."""

    def __init__(self, minx, miny, maxx, maxy):
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

    def overlap(self, other):
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

    def __str__(self):
        return f"Bbox(minx={self.minx:.4f}, miny={self.miny:.4f}, maxx={self.maxx:.4f}, maxy={self.maxy:.4f})"


class ProcessedInstance:
    """Contains a processed instance that is detected by the model. Contains the score the algorithm gave, a polygon for the object,
    a bounding box and a local mask (a boolean mask of the size of the bounding box)

    If compression is enabled, then instance masks are automatically stored as memory-efficient objects. Currently two options are
    possible, either using the coco API ('coco') or scipy sparse arrays ('sparse').
    """

    def __init__(
        self,
        score,
        bbox,
        class_index,
        compress="coco",
        image_shape=None,
        global_polygon=None,
        local_mask=None,
    ):
        """Initializes the instance

        Args:
            score (float): score given to the instance
            bbox (Bbox): the bounding box of the object
            class_index (int): the class index of the object
            compress (bool): store masks as RLE
            polygon (MultiPolygon): a shapely MultiPolygon describing the segmented object
            mask (array): local 2D binary mask for the instance
        """
        self.score = float(score)
        self.bbox = bbox
        self.compress = compress
        self._local_mask = None
        self.class_index = class_index
        self.image_shape = image_shape

        # For most cases, we only need to store the mask:
        if local_mask is not None:
            self.local_mask = local_mask

        # If a polygon is supplied store it, but default None
        self._polygon = global_polygon

    def _compress(self, mask):
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

    def _decompress(self, mask):
        if self.compress is None:
            return mask
        elif self.compress == "coco":
            return coco_mask.decode(mask)
        elif self.compress == "sparse":
            return mask.toarray()
        else:
            raise NotImplementedError(
                f"{self.compress} is not a valid compression method"
            )

    @property
    def local_mask(self):
        if self._local_mask is None:

            assert self._polygon is not None

            local_polygon = translate(
                self._polygon, xoff=self.bbox.minx, yoff=self.bbox.miny
            )
            self.local_mask = polygon_to_mask(local_polygon, shape=self.image_shape)[
                self.bbox.miny : self.bbox.height, self.bbox.minx : self.bbox.width
            ]

        return self._decompress(self._local_mask)

    @local_mask.setter
    def local_mask(self, local_mask):
        self._local_mask = self._compress(local_mask)

    @property
    def polygon(self):
        if self._polygon is None:

            assert self._local_mask is not None

            self._create_polygon(self._local_mask)

        return self._polygon

    @polygon.setter
    def polygon(self, mask):
        polygon = mask_to_polygon(mask)
        self._polygon = translate(polygon, xoff=-self.bbox.minx, yoff=-self.bbox.miny)

    def get_pixels(self, image):
        """Gets the pixel values of the image at the location of the object

        Args:
            image (np.array(int)): image

        Returns:
            np.array(int): pixel values at the location of the object
        """
        return image[self.bbox.miny : self.bbox.maxy, self.bbox.minx : self.bbox.maxx][
            self.local_mask
        ]

    @classmethod
    def from_coco_dict(self, annotation):

        score = annotation["score"]

        minx, miny, width, height = annotation["bbox"]
        bbox = Bbox(minx, miny, minx + width, miny + height)

        class_index = annotation["category_id"]

        """
        if annotation['is_crowd'] == 1:
            annotation_mask = mask.decode(annotation['segmentation'])
            self.polygon = mask_to_polygon(annotation_mask)    
        else:
            coords = [(p[0],p[1]) for p in annotation['segmentation'][polygon]]
            annotation_mask = rasterio.features.rasterize(
                                    [self.polygon], out_shape=shape_local_mask
                                ).astype(bool)
        """

        local_mask = coco_mask.decode(annotation["segmentation"])

        return self(score, bbox, class_index, mask=local_mask)

    def to_coco_dict(self, image_shape, image_id=0, instance_id=0):
        annotation = {}
        annotation["id"] = instance_id
        annotation["image_id"] = image_id
        annotation["category_id"] = int(self.class_index)
        annotation["score"] = float(self.score)
        annotation["bbox"] = [
            float(self.bbox.minx),
            float(self.bbox.miny),
            float(self.bbox.width),
            float(self.bbox.height),
        ]
        annotation["area"] = float(self.bbox.area)
        annotation["segmentation"] = {}
        annotation["segmentation"]["size"] = image_shape

        # Store as RLE for simplicity
        annotation["iscrowd"] = 1
        full_mask = np.zeros(image_shape, dtype=bool)
        full_mask[
            self.bbox.miny : self.bbox.miny + self.local_mask.shape[0],
            self.bbox.minx : self.bbox.minx + self.local_mask.shape[1],
        ] = self.local_mask
        annotation["segmentation"]["counts"] = coco_mask.encode(
            np.asfortranarray(full_mask)
        )

        return annotation

    def __str__(self):
        return f"ProcessedInstance(score={self.score:.4f}, class={self.class_index}, {str(self.bbox)})"


class ProcessedResult:
    """A processed result of a model. It contains all trees separately and also a global tree mask, canopy mask and image"""

    def __init__(self, image, instances=[]):
        """Initializes the Processed Result

        Args:
            image (np.array(int)): the image
            instances (List[ProcessedInstance], optional): List of all trees. Defaults to []].
        """
        self.image = image
        self.instances = instances

        self.canopy_mask = self._generate_mask(0)
        self.tree_mask = self._generate_mask(1)

    def visualise(
        self, color_trees=(0.8, 0, 0), color_canopy=(0, 0, 0.8), alpha=0.4, **kwargs
    ):
        """Visualizes the result

        Args:
            color_trees (tuple, optional): rgb value of the trees. Defaults to (0.8, 0, 0).
            color_canopy (tuple, optional): rgb value of the canopy. Defaults to (0, 0, 0.8).
            alpha (float, optional): alpha value. Defaults to 0.4.
        """
        fig, ax = plt.subplots(**kwargs)
        plt.axis("off")

        self.vis_image = self.image.read().transpose(1, 2, 0)

        ax.imshow(self.vis_image)

        canopy_mask_image = np.zeros(
            (self.vis_image.shape[0], self.vis_image.shape[1], 4), dtype=float
        )
        canopy_mask_image[self.canopy_mask == 1] = list(color_canopy) + [alpha]
        ax.imshow(canopy_mask_image)

        tree_mask_image = np.zeros(
            (self.vis_image.shape[0], self.vis_image.shape[1], 4), dtype=float
        )
        tree_mask_image[self.tree_mask == 1] = list(color_trees) + [alpha]
        ax.imshow(tree_mask_image)

        plt.show()

    def _generate_mask(self, class_id):
        """_summary_

        Args:
            class_id:
        """

        mask = np.zeros((self.image.height, self.image.width), dtype=np.uint8)

        for instance in self.instances:
            if instance.class_index == class_id:

                mask[
                    instance.bbox.miny : instance.bbox.miny + instance.bbox.height,
                    instance.bbox.minx : instance.bbox.minx + instance.bbox.width,
                ] = (
                    instance.local_mask * instance.score * 255
                )

        return mask

    def serialise(self, output_folder, overwrite=True, image_path=None):

        logger.info(f"Serialising results to {output_folder}")

        os.makedirs(output_folder, exist_ok=overwrite)

        # Save masks
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
                        "weight": out_width,
                        "compress": "packbits",
                        "count": 1,
                        "nodata": 0,
                        "transform": out_transform,
                        "crs": out_crs,
                    }
                )

            with rasterio.open(
                os.path.join(output_folder, "tree_mask.tif"), "w", **out_meta
            ) as dest:
                dest.write(self.tree_mask, indexes=1)

            with rasterio.open(
                os.path.join(output_folder, "canopy_mask.tif"), "w", **out_meta
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

        output_path = os.path.join(output_folder, "results.json")

        categories = {}
        categories[0] = "canopy"
        categories[1] = "tree"

        dump_instances_coco(
            output_path,
            instances=self.instances,
            image_path=image_path,
            categories=categories,
        )

    def __str__(self) -> str:
        canopy_cover = np.count_nonzero(self.canopy_mask) / np.prod(
            self.image.shape[:2]
        )
        tree_cover = np.count_nonzero(self.tree_mask) / np.prod(self.image.shape[:2])
        return f"ProcessedResult(n_trees={len(self.trees)}, canopy_cover={canopy_cover:.4f}, tree_cover={tree_cover:.4f})"


class PostProcessor:
    """Processes the result of the modelRunner"""

    def __init__(self, config, image=None):
        """Initializes the PostProcessor

        Args:
            config (DotMap): the configuration
            image (DatasetReader): input rasterio image
            threshold (float, optional): threshold for adding the detected objects. Defaults to 0.5.
        """
        self.config = config
        # TODO: set the threshold via config for traceable experiments?
        self.threshold = config.postprocess.confidence_threshold
        self.cache_folder = config.postprocess.output_folder

        if self.config.postprocess.stateful:
            os.makedirs(self.cache_folder, exist_ok=True)
            logger.info(f"Caching to {self.cache_folder}")

        if image is not None:
            self.initialise(image)

    def initialise(self, image):
        self.untiled_instances = []
        self.image = image
        self.tile_count = 0

    def _get_proper_bbox(self, bbox=None):
        """Gets the proper bbox of an image given a Detectron Bbox

        Args:
            bbox (Detectron.BoundingBox): Original bounding box of the detectron algorithm. Defaults to None (bbox is entire image)

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

    def non_max_suppression(self, instances, class_index, iou_threshold=0.8):
        """Perform non-maximum suppression on the list of input instances

        Args:
            instances (list(ProcessedInstance)): instances to filter
            class_index (int): class of interest
            iou_threshold (float, optional): IOU threshold Defaults to 0.8.

        Returns:
            list(int): List of indices of boxes to keep
        """

        boxes = []
        scores = []
        all_indices = []

        for idx, instance in enumerate(instances):

            if instance.class_index != class_index:
                continue

            x1, x2 = float(instance.bbox.minx), float(instance.bbox.maxx)
            y1, y2 = float(instance.bbox.miny), float(instance.bbox.maxy)

            boxes.append([x1, x2, y1, y2])
            scores.append(instance.score)
            all_indices.append(idx)

        if len(boxes) > 0:

            all_indices = np.array(all_indices)
            boxes = np.array(boxes, dtype=np.float32)

            scores = torch.Tensor(scores)
            boxes = torch.from_numpy(boxes)

            keep_indices = torchvision.ops.nms(boxes, scores, iou_threshold)
            return all_indices[keep_indices]

        else:
            return []

    def remove_edge_predictions(self):
        pass

    def process_untiled_result(self, result):
        """Processes results outputted by Detectron without tiles

        Args:
            results (Instances): Results predicted by the detectron model

        Returns:
            ProcessedResult: ProcessedResult of the segmentation task
        """
        return self.process_tiled_result([[result, None]])

    def detectron_to_instance(self, result):

        tstart = time.time()

        instances, bbox = result
        proper_bbox = self._get_proper_bbox(bbox)
        out = []

        for instance_index in range(len(instances)):
            class_idx = int(instances.pred_classes[instance_index])

            bbox_instance_tiled = (
                instances.pred_boxes[instance_index].tensor[0].int().cpu().numpy()
            )

            bbox_instance = Bbox(
                minx=proper_bbox.minx + bbox_instance_tiled[0],
                miny=proper_bbox.miny + bbox_instance_tiled[1],
                maxx=proper_bbox.minx + bbox_instance_tiled[2],
                maxy=proper_bbox.miny + bbox_instance_tiled[3],
            )

            global_mask = instances.pred_masks[instance_index].cpu().numpy()
            local_mask = global_mask[
                bbox_instance_tiled[1] : bbox_instance_tiled[3],
                bbox_instance_tiled[0] : bbox_instance_tiled[2],
            ]

            new_instance = ProcessedInstance(
                class_index=class_idx,
                local_mask=local_mask,
                bbox=bbox_instance,
                image_shape=self.image.shape,
                score=instances.scores[instance_index],
                compress="sparse",
            )

            out.append(new_instance)

        telapsed = time.time() - tstart
        logger.debug(f"Processed instances {len(instances)} in {telapsed:1.2f}s")

        return out

    def cache_tiled_result(self, result):

        processed_instances = self.detectron_to_instance(result)

        categories = {}
        categories[0] = "canopy"
        categories[1] = "tree"

        self.tile_count += 1

        dump_instances_coco(
            os.path.join(self.cache_folder, f"{self.tile_count}_instances.json"),
            instances=processed_instances,
            image_path=self.image.name,
            categories=categories,
        )

    def process_cached(self):

        logger.info(
            f"Looking for cached files in: {os.path.abspath(self.cache_folder)}"
        )

        cache_files = glob(os.path.join(self.cache_folder, f"*_instances.json"))
        self.untiled_instances = []

        for cache_file in tqdm(cache_files):

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
                instance = ProcessedInstance.from_coco_dict(annotation)
                self.untiled_instances.append(instance)

    def append_tiled_result(self, result):

        processed_instances = self.detectron_to_instance(result)
        self.untiled_instances.extend(processed_instances)

        self.tile_count += 1

    def _collect_tiled_result(self, results):
        """Collects all segmented objects that are predicted and puts them in a ProcessedResult. Also creates global masks for trees and canopies

        Args:
            results (List[[Instances, Detectron.BoundingBox]]): Results predicted by the detectron model
            threshold (float, optional): threshold for adding the detected objects. Defaults to 0.5.

        """

        for result in results:
            self.append_tiled_result(result)

    def process_tiled_result(self, results=None):
        """Processes the result of the detectron model when the tiled version was used

        Args:
            results (List[[Instances, Detectron.BoundingBox]]): Results predicted by the detectron model. Defaults to None.

        Returns:
            ProcessedResult: ProcessedResult of the segmentation task
        """

        logger.info("Collecting results")

        assert self.image is not None

        if results is not None:
            self._collect_tiled_result(results)

        self.merged_instances = []

        logger.info("Running non-max suppression")

        tree_indices = self.non_max_suppression(
            self.untiled_instances,
            class_index=1,
            iou_threshold=self.config.postprocess.iou_threshold,
        )

        for idx in tree_indices:
            self.merged_instances.append(self.untiled_instances[idx])

        for instance in self.untiled_instances:
            if instance.class_index == 0:
                self.merged_instances.append(instance)

        logger.info("Result collection complete")

        return ProcessedResult(image=self.image, instances=self.merged_instances)
