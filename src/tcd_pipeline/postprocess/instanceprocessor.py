import logging
import time
from typing import Any, Optional, Union

import numpy as np
import rasterio

from tcd_pipeline.cache import COCOInstanceCache, PickleInstanceCache
from tcd_pipeline.result.instancesegmentationresult import InstanceSegmentationResult
from tcd_pipeline.util import Bbox, Vegetation

from .postprocessor import PostProcessor
from .processedinstance import ProcessedInstance, non_max_suppression

logger = logging.getLogger(__name__)


class InstanceSegmentationPostProcessor(PostProcessor):
    def __init__(self, config: dict, image: Optional[rasterio.DatasetReader] = None):
        """Initializes the PostProcessor

        Args:
            config (DotMap): the configuration
            image (DatasetReaer): input rasterio image
        """
        super().__init__(config, image)
        self.cache_suffix = "instance"

    def setup_cache(self):
        cache_format = self.config.postprocess.cache_format

        if cache_format == "coco":
            self.cache = COCOInstanceCache(
                self.cache_folder,
                self.image.name,
                self.config.data.classes,
                self.cache_suffix,
            )
        elif cache_format == "pickle":
            self.cache = PickleInstanceCache(
                self.cache_folder,
                self.image.name,
                self.config.data.classes,
                self.cache_suffix,
            )
        else:
            raise NotImplementedError(f"Cache format {cache_format} is unsupported")

    def detectron_to_instances(
        self,
        predictions,
        tile_bbox,
        edge_tolerance: Optional[int] = 5,
    ) -> list[ProcessedInstance]:
        """Convert a Detectron2 result to a list of ProcessedInstances

        Args:
            predictions (tuple[Instance, Bbox]): result containing the Detectron instances and the bounding box
            of the tile
            tile_bbox: the tile bounding box
            edge_tolerance (int): threshold to remove trees at the edge of the image, not applied to canopy

        Returns
            list[ProcessedInstance]: list of instances

        """

        tstart = time.time()
        instances = predictions
        out = []

        for instance_index in range(len(instances)):
            class_idx = int(instances.pred_classes[instance_index])

            bbox_instance_tiled = (
                instances.pred_boxes[instance_index].tensor[0].int().cpu().numpy()
            )

            global_mask = instances.pred_masks[instance_index].cpu().numpy()
            pred_height, pred_width = global_mask.shape

            bbox_instance = Bbox(
                minx=tile_bbox.minx + max(0, bbox_instance_tiled[0]),
                miny=tile_bbox.miny + max(0, bbox_instance_tiled[1]),
                maxx=tile_bbox.minx + min(pred_width, bbox_instance_tiled[2]),
                maxy=tile_bbox.miny + min(pred_height, bbox_instance_tiled[3]),
            )

            # Filter tree boxes that touch the edge of the tile
            # TODO change to check thing classes
            if class_idx == Vegetation.TREE:
                if bbox_instance_tiled[0] < edge_tolerance:
                    continue
                if bbox_instance_tiled[1] < edge_tolerance:
                    continue
                if bbox_instance_tiled[2] > (pred_height - edge_tolerance):
                    continue
                if bbox_instance_tiled[3] > (pred_width - edge_tolerance):
                    continue

            local_mask = global_mask[
                bbox_instance_tiled[1] : bbox_instance_tiled[3],
                bbox_instance_tiled[0] : bbox_instance_tiled[2],
            ]

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

    def _transform(self, result: dict) -> dict:
        out = {}
        out["instances"] = self.detectron_to_instances(result["predictions"])

        return out

    def cache_result(self, result: dict) -> None:
        """Cache a single tile result

        Args:
            result (tuple[Instance, Bbox]): result containing the Detectron instances and the bounding box
            of the tile

        """
        preds, bbox = result["predictions"], result["bbox"]
        instances = self.detectron_to_instances(preds, bbox)
        self.cache.save(instances, bbox)

    # TODO: Handle merging instances again
    def merge(
        self,
        instance: ProcessedInstance,
        instance_tile_id: int,
        iou_threshold: Optional[int] = 0.2,
    ) -> ProcessedInstance:
        """Merges instances from other tiles if they overlap

        Args:
            new_result (ProcessedInstanace): Instance from the tile that is currently added
            tile_instance (int): Tile from which the instance is
            iou_threshold (Optional[int], optional): Threshold for merging. Defaults to 0.2.

        Returns:
            ProcessedInstance: The instance that needs to be added to the list
        """
        for tile_id, tile in enumerate(self.results[:instance_tile_id]):
            # Only consider tiles that overlap the current one
            if tile["bbox"].overlap(self.results[instance_tile_id]["bbox"]):
                for i, other_instance in enumerate(self.results[tile_id]["instances"]):
                    if (
                        other_instance.class_index == instance.class_index
                        and other_instance.bbox.overlap(instance.bbox)
                    ):
                        intersection = other_instance.polygon.intersection(
                            instance.polygon
                        )
                        union = other_instance.polygon.union(instance.polygon)
                        iou = intersection.area / (union.area + 1e-9)
                        if iou > iou_threshold:
                            new_bbox = Bbox(
                                minx=min(other_instance.bbox.minx, instance.bbox.minx),
                                miny=min(other_instance.bbox.miny, instance.bbox.miny),
                                maxx=max(other_instance.bbox.maxx, instance.bbox.maxx),
                                maxy=max(other_instance.bbox.maxy, instance.bbox.maxy),
                            )

                            # Check if we have class scores
                            if (
                                instance.class_scores is not None
                                and other_instance.class_scores is not None
                            ):
                                _new_score = instance.class_scores
                                _other_score = other_instance.class_scores
                            else:
                                _new_score = instance.score
                                _other_score = other_instance.score

                            # Scale relative to join polygon areas
                            new_score = (
                                _new_score * instance.polygon.area
                                + _other_score * other_instance.polygon.area
                            )
                            new_score /= (
                                instance.polygon.area + other_instance.polygon.area
                            )

                            self.results[instance_tile_id]["instances"][i].update(
                                score=new_score,
                                bbox=new_bbox,
                                class_index=instance.class_index,
                                compress=instance.compress,
                                global_polygon=union,
                                label=other_instance.label,
                            )

                            return self.results[instance_tile_id]["instances"][i]
        return instance

    def process(self) -> InstanceSegmentationResult:
        """Processes the result of the detectron model when the tiled version was used

        Args:
            results (List[[Instances, BoundingBox]]): Results predicted by the detectron model. Defaults to None.

        Returns:
            ProcessedResult: ProcessedResult of the segmentation task
        """

        logger.debug("Collecting results")
        assert self.image is not None

        if self.config.postprocess.stateful:
            self.cache.load()
            self.results = self.cache.results

        # Combine instances from individual results
        self.all_instances = set()

        for tile in self.results:
            for instance in tile["instances"]:
                self.all_instances.add(instance)

        self.all_instances = list(self.all_instances)

        # Optionally run NMS
        self.merged_instances = []

        if self.config.postprocess.use_nms:
            logger.info("Running non-max suppression")

            nms_indices = non_max_suppression(
                self.all_instances,
                class_index=Vegetation.TREE,
                iou_threshold=self.config.postprocess.iou_threshold,
            )

            if isinstance(nms_indices, np.int64):
                nms_indices = [nms_indices]

            for idx in nms_indices:
                self.merged_instances.append(self.all_instances[idx])

            nms_indices = non_max_suppression(
                self.all_instances,
                class_index=Vegetation.CANOPY,
                iou_threshold=self.config.postprocess.iou_threshold,
            )

            if isinstance(nms_indices, np.int64):
                nms_indices = [nms_indices]

            for idx in nms_indices:
                self.merged_instances.append(self.all_instances[idx])
        else:
            self.merged_instances = self.all_instances

        logger.debug("Result collection complete")

        return InstanceSegmentationResult(
            image=self.image, instances=self.merged_instances, config=self.config
        )
