import logging
import time
from typing import Any, Optional, Union

import numpy as np
import rasterio
from detectron2.structures import Instances
from shapely.geometry import box

from tcd_pipeline.cache import PickleInstanceCache, ShapefileInstanceCache
from tcd_pipeline.result.instancesegmentationresult import InstanceSegmentationResult
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
        self.cache_suffix = "instance"

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

    def dissolve(self, instances: list[ProcessedInstance], buffer: int = -5) -> dict:
        from collections import defaultdict

        import matplotlib.pyplot as plt
        import shapely
        from rtree import index
        from shapely.plotting import plot_polygon

        # Unary union gives a single geometry - split it into polygons:
        union = shapely.unary_union([i.polygon.buffer(buffer) for i in instances])

        if isinstance(union, shapely.geometry.Polygon):
            merged = [union]
        else:
            merged = [g for g in union.geoms]

        out = defaultdict(list)

        # Add the merged/dissolved polygons to a spatial index
        idx = index.Index()

        for i, m_poly in enumerate(merged):
            idx.insert(i, m_poly.bounds, obj=m_poly)

        # Iterate over the source instances and associate source <> merged
        for instance in instances:
            poly = instance.polygon
            if np.any(np.isnan(poly.bounds)):
                continue
            potential_candidates = idx.intersection(poly.bounds, objects=True)
            for n in potential_candidates:
                if poly.intersects(n.object):
                    out[n.object].append(instance)

        return out

    def merge(
        self,
        instances: list[ProcessedInstance],
        class_index: int,
    ) -> list[ProcessedInstance]:
        """Merges instances from other tiles if they overlap

        Args:
            instances (list[ProcessedInstance]): Instance list to consider merging
            class_index (int): Class filter
        Returns:
            list[ProcessedInstance]: List of merged instances
        """
        import shapely

        tiles = [tile["bbox"] for tile in self.results]

        # Keep track of which tiles we've compared
        merged_tiles = set()

        # Polygons in outer regions
        merged_instances = set()
        neighbours = find_overlapping_neighbors(tiles)

        # Keep track of all instances
        tile_instances = {}
        for tile_idx, tile in enumerate(tiles):
            tile_instances[tile_idx] = set(self.results[tile_idx]["instances"])

        merge = set()

        for tile_idx, tile in enumerate(tiles):
            other_class = set()

            for neighbour_idx in neighbours[tile_idx]:
                # Skip already processed pairs
                if (neighbour_idx, tile_idx) in merged_tiles:
                    continue
                else:
                    merged_tiles.add((neighbour_idx, tile_idx))
                    merged_tiles.add((tile_idx, neighbour_idx))

                # Get the overlap between the tile and this neighbour
                neighbour = tiles[neighbour_idx]
                overlap_region = tile.intersection(neighbour)

                # Instances to merge from the current tile
                for instance in tile_instances[tile_idx]:
                    if not instance.class_index == class_index:
                        other_class.add(instance)
                    elif instance.polygon.intersects(overlap_region):
                        merge.add(instance)

                tile_instances[tile_idx] = tile_instances[tile_idx].difference(
                    merge, other_class
                )

                # Instances to merge fromt the neighbouring tile
                for instance in tile_instances[neighbour_idx]:
                    if not instance.class_index == class_index:
                        other_class.add(instance)
                    elif instance.polygon.intersects(overlap_region):
                        merge.add(instance)

                tile_instances[neighbour_idx] = tile_instances[
                    neighbour_idx
                ].difference(merge, other_class)

        for poly, instances in self.dissolve(merge, buffer=-5).items():
            # Re-pad polygons after dissolve
            poly = poly.buffer(5)

            # TODO Check what to do a merged instance contains many polygons. Buffering helps here
            # somewhat, but it's not perfect.

            new_instance = ProcessedInstance(
                score=np.median([i.score for i in instances]),
                bbox=shapely.geometry.box(*poly.bounds),
                global_polygon=poly,
                class_index=class_index,
            )

            merged_instances.add(new_instance)

        non_merged_instances = set.union(*[s for s in tile_instances.values()])
        for poly, instances in self.dissolve(non_merged_instances, buffer=-5).items():
            # Re-pad polygons after dissolve
            poly = poly.buffer(5)
            new_instance = ProcessedInstance(
                score=np.median([i.score for i in instances]),
                bbox=shapely.geometry.box(*poly.bounds),
                global_polygon=poly,
                class_index=class_index,
            )

            merged_instances.add(new_instance)

        return list(merged_instances)

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

        return InstanceSegmentationResult(
            image=self.image, instances=self.merged_instances, config=self.config
        )
