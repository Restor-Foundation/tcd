import logging
import time
from collections import defaultdict
from typing import Any, Optional, Union

import numpy as np
import rasterio
import rtree
import shapely
from detectron2.structures import Instances
from shapely.geometry import box
from tqdm.auto import tqdm

from tcd_pipeline.cache import PickleInstanceCache, ShapefileInstanceCache
from tcd_pipeline.result.instancesegmentationresult import (
    InstanceSegmentationResult,
    save_shapefile,
)
from tcd_pipeline.util import Vegetation, find_overlapping_neighbors, inset_box

from .postprocessor import PostProcessor
from .processedinstance import ProcessedInstance, non_max_suppression

logger = logging.getLogger(__name__)


class InstanceSegmentationPostProcessor(PostProcessor):
    def __init__(self, config: dict, image: Optional[rasterio.DatasetReader] = None):
        """Initializes the PostProcessor

        Args:
            config (dict): the configuration
            image (rasterio.DatasetReader, optional): input rasterio image
        """
        super().__init__(config, image)
        self.cache_suffix = None

    def setup_cache(self):
        cache_format = self.config.postprocess.cache_format

        if cache_format == "pickle":
            self.cache = PickleInstanceCache(
                self.cache_folder,
                self.image.name,
                self.config.data.classes,
                self.cache_suffix,
            )
        elif cache_format == "shapefile":
            self.cache = ShapefileInstanceCache(
                self.cache_folder,
                self.image.name,
                self.config.data.classes,
                self.cache_suffix,
            )
        else:
            raise NotImplementedError(f"Cache format {cache_format} is unsupported")

    def detectron_to_instances(
        self,
        predictions: Instances,
        tile_bbox: box,
        edge_tolerance: Optional[int] = 5,
    ) -> list[ProcessedInstance]:
        """Convert a Detectron2 result to a list of ProcessedInstances

        Args:
            predictions (Instances): Detectron instances
            tile_bbox (shapely.geometry.box): the tile bounding box
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

            minx, miny, _, _ = tile_bbox.bounds

            bbox_instance = box(
                minx=minx + max(0, bbox_instance_tiled[0]),
                miny=miny + max(0, bbox_instance_tiled[1]),
                maxx=minx + min(pred_width, bbox_instance_tiled[2]),
                maxy=miny + min(pred_height, bbox_instance_tiled[3]),
            )
            assert bbox_instance.intersects(tile_bbox)

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
            result (dict): result containing the Detectron instances and the bounding box
                           of the tile

        """
        preds, bbox = result["predictions"], result["bbox"]
        instances = self.detectron_to_instances(preds, bbox)
        self.cache.save(instances, bbox)

    def dissolve(self, instances: list[ProcessedInstance]) -> dict:
        """
        Perform a dissolve operation and return a dictionary of merged
        polygons and their sub-polygons (i.e. from the source list).

        Returns a dictionary where the keys are dissolved polygons and
        the values are lists of source geometries that intersected it.

        Args:
            instances (list[ProcessedInstance]): Instance list
        Returns:
            dict [shapely.Polygon -> list[ProcessedInstance]: Mapping between dissolved geometry and source geometries

        """

        # Unary union gives a single geometry - split it into polygons:
        union = shapely.unary_union([i.polygon for i in instances])
        # Case when the union is a single polygon
        if isinstance(union, shapely.geometry.Polygon):
            merged = [union]
        else:
            merged = [g for g in union.geoms]

        out = defaultdict(list)

        # Add the merged/dissolved polygons to a spatial index
        idx = rtree.index.Index()

        for i, m_poly in enumerate(merged):
            idx.insert(i, m_poly.bounds, obj=m_poly)

        used_geoms = set()

        # Iterate over the source instances and associate source <> merged
        for instance in instances:
            poly = instance.polygon
            if np.any(np.isnan(poly.bounds)):
                continue
            # BBOX intersection
            potential_candidates = idx.intersection(poly.bounds, objects=True)
            for n in potential_candidates:
                # Polygon intersection
                if poly.intersects(n.object) and instance.polygon not in used_geoms:
                    out[n.object].append(instance)
                    used_geoms.add(instance.polygon)

        return out

    def filter_centroids(
        self, instances: list[ProcessedInstance], max_overlaps: int = 1
    ) -> list[ProcessedInstance]:
        """
        Filter objects to remove those which contain the centroids
        of others - this is a simple but fairly effective heuristic
        to remove "dumbell" shaped predictions which contain multiple
        individuals.

        Args:
            instances (list[ProcessedInstance]): Instance list to consider merging
            max_overlaps (int): Number of centroids that a polygon can contain to be
                                considered non-overlapping (default 1)
        Returns:
            list[ProcessedInstance]: List of merged instances

        """
        instances = list(instances)

        overlaps = defaultdict(int)

        for a in instances:
            for idx, b in enumerate(instances):
                if a == b:
                    continue
                if a.polygon.centroid.within(b.polygon):
                    overlaps[idx] += 1

        out = []
        for idx, a in enumerate(instances):
            if overlaps[idx] > max_overlaps:
                continue

            out.append(a)

        return out

    def split(
        self, instances: list[ProcessedInstance], iou_threshold=0.3
    ) -> list[ProcessedInstance]:
        """
        Split a list of instances based on heuristics.

        First, instancs are filtered based on whether they
        contain centroids of other instances. Then, any
        instances which have a small overlap are considered
        separate. Instances with a largeer overlap are merged.

        This process continues iteratively until all instances
        have been either merged or split out.

        Args:
            instances (list[ProcessedInstance]): Instance list to consider merging
            iou_threshold(float): Threshold above which to consider a polygon as overlapping
        Returns:
            list[ProcessedInstance]: List of merged instances
        """
        instances = list(instances)
        merged = []

        instances = self.filter_centroids(instances)

        while len(instances) > 0:
            instance_a = instances.pop()
            a_overlaps = False
            ab_union = None

            for idx, instance_b in enumerate(instances):
                if instance_a == instance_b:
                    continue

                ab_intersection = instance_a.polygon.intersection(instance_b.polygon)
                ab_union = instance_a.polygon.union(instance_b.polygon)
                iou = ab_intersection.area / ab_union.area

                if iou < iou_threshold:
                    continue
                else:
                    a_overlaps = True
                    break

            # Base case, no significant overlap
            if not a_overlaps:
                merged.append(instance_a)
            # Otherwise, we should combine a, b
            # and add that polygon back to the
            # instance list.
            else:
                instances.pop(idx)
                instances.append(instance_a + instance_b)

        return merged

    def merge(
        self,
        instances: list[ProcessedInstance],
        class_index: int,
        confidence_threshold: float = 0.4,
        iou_threshold: float = 0.5,
    ) -> list[ProcessedInstance]:
        """Merge a list of instances.

        1) Filter by confidence threshold
        2) Dissovle instances to separate overlapping groups
        3) For each group, split into polygons following simple heuristics
           including IoU and centroid overlap

        Args:
            instances (list[ProcessedInstance]): Instance list to consider merging
            class_index (int): Class filter
            confidence_threshold (float): Confidence threshold
            iou_threshold(float): Threshold above which to consider a polygon as overlapping
        Returns:
            list[ProcessedInstance]: List of merged instances
        """

        merged_instances = []
        instances = list(
            filter(
                lambda x: x.score > confidence_threshold
                and x.class_index == class_index,
                instances,
            )
        )

        for _, instance_group in tqdm(self.dissolve(instances).items()):
            if len(instance_group) == 1:
                instance = instance_group[0]
                merged_instances.append(instance)
            else:
                split_instances = self.split(
                    instance_group, iou_threshold=iou_threshold
                )
                for instance in split_instances:
                    if isinstance(instance.score, list):
                        instance.score = np.median(instance.score)

                    merged_instances.append(instance)

        return merged_instances

    def process(self) -> InstanceSegmentationResult:
        """Processes the result of the detectron model when the tiled version was used

        Returns:
            InstanceSegmentationResult for the job
        """

        logger.debug("Collecting results")
        assert self.image is not None

        if isinstance(self.cache, ShapefileInstanceCache):
            import shutil

            shutil.copytree(
                self.cache.cache_folder, self.config.data.output, dirs_exist_ok=True
            )

        if self.config.postprocess.stateful:
            logger.info("Loading results from cache")
            self.cache.load()
            self.results = self.cache.results

        # Combine instances from individual results
        self.all_instances = set()

        for idx, result in enumerate(self.results):
            for instance in result["instances"]:
                self.all_instances.add(instance)

        self.all_instances = list(self.all_instances)

        # Optionally run NMS
        if self.config.postprocess.use_nms:
            logger.info("Running non-max suppression")
            self.merged_instances = []

            for class_index in [Vegetation.TREE, Vegetation.CANOPY]:
                nms_indices = non_max_suppression(
                    self.all_instances,
                    class_index=class_index,
                    iou_threshold=self.config.postprocess.iou_threshold,
                )

                if isinstance(nms_indices, np.int64):
                    nms_indices = [nms_indices]

                for idx in nms_indices:
                    self.merged_instances.append(self.all_instances[idx])

        else:
            self.merged_instances = self.all_instances

        if self.config.postprocess.dissolve:
            logger.info("Dissolving remaining polygons")
            self.merged_trees = self.merge(
                self.merged_instances, class_index=Vegetation.TREE
            )
            self.merged_canopy = self.merge(
                self.merged_instances, class_index=Vegetation.CANOPY
            )
            self.merged_instances = self.merged_trees + self.merged_canopy

        logger.debug("Result collection complete")

        import os

        save_shapefile(
            self.merged_instances,
            output_path=os.path.join(
                self.config.data.output, "instances_processed.shp"
            ),
            image=self.image,
            include_bbox=None,
        )

        return InstanceSegmentationResult(
            image=self.image, instances=self.merged_instances, config=self.config
        )
