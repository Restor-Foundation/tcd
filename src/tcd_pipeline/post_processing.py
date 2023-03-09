import json
import logging
import os
import pickle
import shutil
import time
from glob import glob
from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt
import rasterio
import torch
import torchvision
from detectron2.structures import Instances
from natsort import natsorted
from torchgeo.datasets import BoundingBox
from tqdm.auto import tqdm

from .instance import ProcessedInstance, dump_instances_coco
from .result import InstanceSegmentationResult, ProcessedResult, SegmentationResult
from .util import Bbox, Vegetation

logger = logging.getLogger(__name__)


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
        self.cache_suffix = "instances"
        self.cache_bboxes_name = "tiled_bboxes"

        if image is not None:
            self.initialise(image)

    def initialise(self, image, warm_start=True) -> None:
        """Initialise the processor for a new image and creates cache
        folders if required.

        Args:
            image (DatasetReader): input rasterio image
            warm_start (bool, option): Whether or not to continue from where one left off. Defaults to True.
        """
        self.untiled_results = []
        self.image = image
        self.tile_count = 0
        self.warm_start = warm_start
        self.tiled_bboxes = []
        self.cache_folder = os.path.abspath(
            os.path.join(
                self.cache_root,
                os.path.splitext(os.path.basename(self.image.name))[0] + "_cache",
            )
        )

        # Always clear the cache directory if we're doing a cold start
        if not warm_start:
            logger.debug(f"Attempting to clear existing cache")
            self.reset_cache()

        else:
            logger.info(f"Attempting to use cached result from {self.cache_folder}")
            # Check to see if we have a bounding box file
            # this stores how many tiles we've processed

            bboxes_path = os.path.join(
                self.cache_folder, f"{self.cache_bboxes_name}.pkl"
            )

            if self.image is not None and os.path.exists(bboxes_path):

                self.tiled_bboxes = self._load_cache_pickle(bboxes_path)
                self.tile_count = len(self._get_cache_tile_files())

                # We should probably have a strict mode that will error out
                # if there's a cache mismatch
                if self.tile_count != len(self.tiled_bboxes):
                    logger.warning(
                        "Missing bounding boxes. Check the temporary folder to ensure this is expected behaviour."
                    )

                    self.tile_count = min(self.tile_count, len(self.tiled_bboxes))
                    self.tiled_bboxes = self.tiled_bboxes[: self.tile_count]

                if self.tile_count > 0:
                    logger.info(f"Starting from tile {self.tile_count + 1}.")

            # Otherwise we should clear the cache
            else:
                self.reset_cache()

    def reset_cache(self):
        """Clear cache. Warning: there are no checks here, the set cache folder and its
        contents will be deleted.
        """

        if self.config.postprocess.stateful:

            if self.cache_folder is None:
                logger.warning("Cache folder not set")
                return

            if not os.path.exists(self.cache_folder):
                logger.warning("Cache folder doesn't exist")
            else:
                shutil.rmtree(self.cache_folder)
                logger.warning(f"Removed existing cache folder {self.cache_folder}")

            os.makedirs(self.cache_folder, exist_ok=True)
            logger.info(f"Caching to {self.cache_folder}")
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

            if instances.has("class_scores"):
                scores = instances.class_scores[instance_index]
            else:
                scores = instances.scores[instance_index]

            new_result = ProcessedInstance(
                class_index=class_idx,
                local_mask=local_mask,
                bbox=bbox_instance,
                score=scores,
                compress="sparse",
            )

            out.append(new_result)

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

        with open(
            os.path.join(self.cache_folder, f"{self.cache_bboxes_name}.pkl"), "wb"
        ) as fp:
            pickle.dump(self.tiled_bboxes, fp)

        if cache_format == "coco":
            dump_instances_coco(
                os.path.join(
                    self.cache_folder, f"{self.tile_count}_{self.cache_suffix}.json"
                ),
                instances=processed_instances,
                image_path=self.image.name,
                categories=categories,
            )
        elif cache_format == "pickle":
            self._save_cache_pickle(processed_instances)
        elif cache_format == "numpy":
            self._save_cache_numpy(processed_instances)
        else:
            raise NotImplementedError(f"Cache format {cache_format} is unsupported")

        if self.config.postprocess.debug_images:
            self.cache_tile_image(result[1])

    def _save_cache_pickle(self, processed_instances):
        with open(
            os.path.join(
                self.cache_folder, f"{self.tile_count}_{self.cache_suffix}.pkl"
            ),
            "wb",
        ) as fp:
            pickle.dump(processed_instances, fp)

    def _save_cache_numpy(self, processed_instances):
        raise NotImplementedError

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
        new_result: ProcessedInstance,
        tile_instance: int,
        iou_threshold: Optional[int] = 0.2,
    ):
        """Merges instances from other tiles if they overlap

        Args:
            new_result (ProcessedInstanace): Instance from the tile that is currently added
            tile_instance (int): Tile from which the instance is
            iou_threshold (Optional[int], optional): Threshold for merging. Defaults to 0.2.

        Returns:
            ProcessedInstance: The instance that needs to be added to the list
        """
        for tile_index, tile in enumerate(self.tiled_bboxes[:tile_instance]):
            if tile.overlap(self.tiled_bboxes[tile_instance]):
                for i, other_instance in enumerate(self.untiled_results[tile_index]):
                    if (
                        other_instance.class_index == new_result.class_index
                        and other_instance.bbox.overlap(new_result.bbox)
                    ):
                        intersection = other_instance.polygon.intersection(
                            new_result.polygon
                        )
                        union = other_instance.polygon.union(new_result.polygon)
                        iou = intersection.area / (union.area + 1e-9)
                        if iou > iou_threshold:
                            new_bbox = Bbox(
                                minx=min(
                                    other_instance.bbox.minx, new_result.bbox.minx
                                ),
                                miny=min(
                                    other_instance.bbox.miny, new_result.bbox.miny
                                ),
                                maxx=max(
                                    other_instance.bbox.maxx, new_result.bbox.maxx
                                ),
                                maxy=max(
                                    other_instance.bbox.maxy, new_result.bbox.maxy
                                ),
                            )

                            # Check if we have class scores
                            if (
                                new_result.class_scores is not None
                                and other_instance.class_scores is not None
                            ):
                                _new_score = new_result.class_scores
                                _other_score = other_instance.class_scores
                            else:
                                _new_score = new_result.score
                                _other_score = other_instance.score

                            # Scale relative to join polygon areas
                            new_score = (
                                _new_score * new_result.polygon.area
                                + _other_score * other_instance.polygon.area
                            )
                            new_score /= (
                                new_result.polygon.area + other_instance.polygon.area
                            )

                            self.untiled_results[tile_index][i].update(
                                score=new_score,
                                bbox=new_bbox,
                                class_index=new_result.class_index,
                                compress=new_result.compress,
                                global_polygon=union,
                                label=other_instance.label,
                            )

                            return self.untiled_results[tile_index][i]
        return new_result

    def merge_results_other_tiles(
        self, new_results: list[ProcessedInstance], tile_results: int
    ):
        """Merges the results from a new tile into the predictions

        Args:
            new_results (list[ProcessedInstance]): List of processed results
            tile_results (int): Tile from which the instances are
        """
        new_annotations = []
        for annotation in new_results:
            if (
                Vegetation(annotation.class_index).name.lower()
                in self.config.postprocess.merge_classes
            ):
                merged = self.merge_instance(
                    annotation, tile_results, self.config.postprocess.iou_tiles
                )
                new_annotations.append(merged)
            else:
                new_annotations.append(annotation)

        self.untiled_results.append(new_annotations)

    def _get_cache_tile_files(self) -> list:

        cache_format = self.config.postprocess.cache_format

        if cache_format == "pickle":
            cache_glob = f"*_{self.cache_suffix}.pkl"
        elif cache_format == "coco":
            cache_glob = f"*_{self.cache_suffix}.json"
        elif cache_format == "numpy":
            cache_glob = f"*_{self.cache_suffix}.npz"
        else:
            raise NotImplementedError(f"Cache format {cache_format} is unsupported")

        cache_files = natsorted(glob(os.path.join(self.cache_folder, cache_glob)))

        if len(cache_files) == 0:
            logger.warning("No cached files found.")

        return cache_files

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

        self.tiled_bboxes = self._load_cache_pickle(
            f"{os.path.join(self.cache_folder, self.cache_bboxes_name)}.pkl"
        )

        cache_files = self._get_cache_tile_files()

        self.untiled_results = []
        cache_format = self.config.postprocess.cache_format

        for i in tqdm(range(len(cache_files))):
            cache_file = cache_files[i]  # no enumerate because uglier tqdm
            if cache_format == "coco":
                annotations = self._load_cache_coco(cache_file)
                logger.debug(f"Loaded {len(annotations)} instances from {cache_file}")
            elif cache_format == "pickle":
                annotations = self._load_cache_pickle(cache_file)
                logger.debug(f"Loaded {len(annotations)} instances from {cache_file}")
            elif cache_format == "numpy":
                annotations = self._load_cache_numpy(cache_file)

            self.merge_results_other_tiles(annotations, i)
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
        self.merge_results_other_tiles(annotations, self.tile_count)

    def _collect_tiled_result(
        self, results: tuple[Union[Instances, torch.Tensor], BoundingBox]
    ) -> None:
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
    ) -> InstanceSegmentationResult:
        """Processes the result of the detectron model when the tiled version was used

        Args:
            results (List[[Instances, BoundingBox]]): Results predicted by the detectron model. Defaults to None.

        Returns:
            ProcessedResult: ProcessedResult of the segmentation task
        """

        logger.debug("Collecting results")

        assert self.image is not None

        if results is not None:
            self._collect_tiled_result(results)

        self.all_instances = []
        for tile in range(len(self.untiled_results)):
            for instance in self.untiled_results[tile]:
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

        return InstanceSegmentationResult(
            image=self.image, instances=self.merged_instances, config=self.config
        )

    def cache_tile_image(self, bbox):
        proper_bbox = self._get_proper_bbox(bbox)

        kwargs = self.image.meta.copy()
        window = proper_bbox.window()

        kwargs.update(
            {
                "height": window.height,
                "width": window.width,
                "transform": rasterio.windows.transform(window, self.image.transform),
                "compress": "jpeg",
            }
        )

        with rasterio.open(
            os.path.join(self.cache_folder, f"{self.tile_count}_tile.tif"),
            "w",
            **kwargs,
        ) as dst:
            dst.write(self.image.read(window=window))


class SegmentationPostProcessor(PostProcessor):
    def __init__(self, config: dict, image: Optional[rasterio.DatasetReader] = None):
        """Initializes the PostProcessor

        Args:
            config (DotMap): the configuration
            image (DatasetReaer): input rasterio image
        """
        super().__init__(config, image)
        self.cache_suffix = "segmentation"

    def append_tiled_result(self, result: tuple[npt.NDArray, BoundingBox]) -> None:
        """
        Adds a tensor result to the processor

        Args:
            result (Any): detectron result
        """
        self.tiled_bboxes.append(self._get_proper_bbox(result[1]))
        self.untiled_results.extend(result)
        self.tile_count += 1

    def _save_cache_pickle(self, result):
        with open(
            os.path.join(
                self.cache_folder, f"{self.tile_count}_{self.cache_suffix}.pkl"
            ),
            "wb",
        ) as fp:

            pickle.dump(result, fp)

    def _save_cache_numpy(self, result):
        file_name = os.path.join(
            self.cache_folder, f"{self.tile_count}_{self.cache_suffix}.npz"
        )

        if result[1] is not None:
            bbox = (
                result[1].minx,
                result[1].maxx,
                result[1].miny,
                result[1].maxy,
                result[1].mint,
                result[1].maxt,
            )
            np.savez(file_name, mask=result[0], bbox=bbox)
        else:
            np.savez(file_name, mask=result[0])

    def cache_tiled_result(self, result: tuple[Instances, BoundingBox]) -> None:

        """Cache a single tile result

        Args:
            result (tuple[Instance, Bbox]): result containing the Detectron instances and the bounding box
            of the tile

        """

        logger.debug(f"Saving cache for tile {self.tile_count}")

        self.tile_count += 1
        self.tiled_bboxes.append(self._get_proper_bbox(result[1]))
        cache_format = self.config.postprocess.cache_format

        with open(
            os.path.join(self.cache_folder, f"{self.cache_bboxes_name}.pkl"), "wb"
        ) as fp:
            pickle.dump(self.tiled_bboxes, fp)

        if cache_format == "pickle":
            self._save_cache_pickle(result)
        elif cache_format == "numpy":
            self._save_cache_numpy(result)
        else:
            raise NotImplementedError(f"Cache format {cache_format} is unsupported")

        if self.config.postprocess.debug_images:
            self.cache_tile_image(result[1])

    def _load_cache_numpy(self, cache_file: str) -> list[npt.NDArray]:
        """Load cached results that were pickled.

        Args:
            cache_file (str): Cache filename

        Returns:
            list[ProcessedInstance]: instances

        """
        data = np.load(cache_file)
        mask = data["mask"]

        if "bbox" in data:

            bbox = BoundingBox(*data["bbox"])
        else:
            bbox = None

        return [[mask, bbox]]

    def merge_results_other_tiles(
        self, new_results: list[ProcessedInstance], tile_number: int
    ):
        self.untiled_results.append(new_results)

    def process_tiled_result(
        self, results: list[tuple[npt.NDArray, BoundingBox]] = None
    ) -> SegmentationResult:
        """Processes the result of the detectron model when the tiled version was used

        Args:
            results (List[[Instances, BoundingBox]]): Results predicted by the detectron model. Defaults to None.

        Returns:
            SegmentationResult: SegmentationResult of the segmentation task
        """

        logger.debug("Collecting results")

        assert self.image is not None

        if results is not None:
            self._collect_tiled_result(results)

        logger.info("Result collection complete")

        return SegmentationResult(
            image=self.image,
            tiled_masks=self.untiled_results,
            bboxes=self.tiled_bboxes,
            config=self.config,
        )
