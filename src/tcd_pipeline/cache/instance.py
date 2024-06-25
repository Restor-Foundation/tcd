import glob
import json
import logging
import os
import pickle
from typing import Dict, List

import fiona
from shapely.geometry import box

from ..postprocess.processedinstance import ProcessedInstance, dump_instances_coco
from .cache import ResultsCache

logger = logging.getLogger(__name__)


class InstanceSegmentationCache(ResultsCache):
    """
    A cache format that can store instance segmentation results. The input
    to, and output from, the cache is a list of ProcessedInstances.
    """

    @property
    def results(self) -> List[dict]:
        """
        A list of results that were stored in the cache folder. Each entry
        in the list is a dictionary containing the following keys:

        - bbox: Bbox
        - image: str
        - instances : List[ProcessedInstance]

        """
        return self._results


class PickleInstanceCache(InstanceSegmentationCache):
    def save(self, instances: List[ProcessedInstance], bbox: box):
        output = {"instances": instances, "bbox": bbox, "image": self.image_path}

        file_name = f"{self.tile_count}_{self.cache_suffix}.pkl"
        output_path = os.path.join(self.cache_folder, file_name)

        with open(output_path, "wb") as fp:
            pickle.dump(output, fp)

        self.tile_count += 1
        self.write_tile_meta(self.tile_count, bbox, output_path)

    def _load_file(self, cache_file: str) -> dict:
        """Load pickled cache results

        Args:
            cache_file (str): Cache filename

        Returns:
            dict: dictionary containing "instances" and "bbox"

        """

        with open(cache_file, "rb") as fp:
            annotations = pickle.load(fp)

        return annotations


class ShapefileInstanceCache(InstanceSegmentationCache):
    def __init__(self, cache_folder, image_path: str, classes=None, cache_suffix=None):
        super().__init__(cache_folder, image_path, classes, cache_suffix)
        self.cache_file = os.path.join(
            self.cache_folder, f"instances{self.cache_suffix}.shp"
        )

    def save(self, instances: List[ProcessedInstance], bbox: box):
        import rasterio

        from tcd_pipeline.result.instancesegmentationresult import save_shapefile

        save_shapefile(
            instances,
            output_path=self.cache_file,
            image=rasterio.open(self.image_path),
            include_bbox=None,
            mode="w" if self.tile_count == 0 else "a",
        )

        self.tile_count += 1
        self.write_tile_meta(self.tile_count, bbox, self.cache_file)

    def _load_file(self, filename) -> List[ProcessedInstance]:
        instances = []

        import rasterio
        import shapely

        with rasterio.open(self.image_path) as src:
            # TODO More efficient/rasterioy way of doing this?
            t = ~src.transform
            transform = [t.a, t.b, t.d, t.e, t.xoff, t.yoff]

            with fiona.open(filename) as cxn:
                for f in cxn:
                    class_index = f["properties"]["class_idx"]
                    score = f["properties"]["score"]

                    try:
                        # World coords
                        polygon = shapely.geometry.shape(f["geometry"])

                        # Image coords
                        global_polygon = shapely.affinity.affine_transform(
                            polygon, transform
                        )

                        bbox = shapely.geometry.box(*global_polygon.bounds)

                        instance = ProcessedInstance(
                            score=score,
                            bbox=bbox,
                            class_index=class_index,
                            global_polygon=global_polygon,
                        )
                        instances.append(instance)
                    # If we can't load an object, try to fail gracefully?
                    except AttributeError:
                        continue

                from rasterio.windows import from_bounds

                w = from_bounds(*cxn.bounds, src.transform)

        results = {
            "instances": instances,
            "bbox": box(w.col_off, w.row_off, w.width, w.height),
        }

        return results


class COCOInstanceCache(InstanceSegmentationCache):
    """
    Cache handler that stores data in MS-COCO format. This is currently only
    used for instance segmentation models, but could in principle be extended
    to support semantic/panoptic segmentation in the future. This method is
    convenient for instance detection because it allows intermediate results
    to be easily inspected using standard annotation tools.
    """

    def save(self, instances: List[ProcessedInstance], bbox: box):
        """
        Save cached results in MS-COCO format.

        Args:
            instances (List[ProcessedInstance]: a list of instances to cache
            bbox: tile bounding box in global image coordinates
        """
        file_name = f"{self.tile_count}{self.cache_suffix}.json"
        output_path = os.path.join(self.cache_folder, file_name)

        metadata = {
            "bbox": bbox.bounds,
            "image": self.image_path,
            "tile_id": self.tile_count,
        }

        dump_instances_coco(
            output_path,
            instances,
            self.image_path,
            categories=self.classes,
            metadata=metadata,
        )

        self.tile_count += 1
        self.write_tile_meta(self.tile_count, bbox, output_path)

    def _load_file(self, cache_file: str) -> dict:
        """Load cached results from MS-COCO format

        Args:
            cache_file (str): Cache filename

        Returns:
            list[ProcessedInstance]: instances

        """

        out = {}
        out["instances"] = []

        with open(cache_file, "r") as fp:
            annotations = json.load(fp)

            out["bbox"] = box(*annotations["metadata"]["bbox"])
            out["image"] = annotations["metadata"]["image"]
            out["tile_id"] = annotations["metadata"]["tile_id"]

            for annotation in annotations["annotations"]:
                out["instances"].append(ProcessedInstance.from_coco_dict(annotation))
        return out
