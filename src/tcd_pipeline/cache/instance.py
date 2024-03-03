import glob
import json
import logging
import os
import pickle
from typing import Dict, List

from tcd_pipeline.util import Bbox

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
    def _find_cache_files(self) -> List[str]:
        return glob.glob(os.path.join(self.cache_folder, f"*_{self.cache_suffix}.pkl"))

    def save(self, instances: List[ProcessedInstance], bbox: Bbox):
        output = {"instances": instances, "bbox": bbox, "image": self.image_path}

        file_name = f"{self.tile_count}_{self.cache_suffix}.pkl"
        output_path = os.path.join(self.cache_folder, file_name)

        with open(output_path, "wb") as fp:
            pickle.dump(output, fp)

        self.tile_count += 1

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


class COCOInstanceCache(InstanceSegmentationCache):
    """
    Cache handler that stores data in MS-COCO format. This is currently only
    used for instance segmentation models, but could in principle be extended
    to support semantic/panoptic segmentation in the future. This method is
    convenient for instance detection because it allows intermediate results
    to be easily inspected using standard annotation tools.
    """

    def _find_cache_files(self) -> List[str]:
        return glob.glob(os.path.join(self.cache_folder, f"*_{self.cache_suffix}.json"))

    def save(self, instances: List[ProcessedInstance], bbox: Bbox):
        """
        Save cached results in MS-COCO format.

        Args:
            instances (List[ProcessedInstance]: a list of instances to cache
            bbox: tile bounding box in global image coordinates
        """
        file_name = f"{self.tile_count}_{self.cache_suffix}.json"
        output_path = os.path.join(self.cache_folder, file_name)

        metadata = {
            "bbox": bbox.bbox,
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

            out["bbox"] = Bbox(*annotations["metadata"]["bbox"])
            out["image"] = annotations["metadata"]["image"]
            out["tile_id"] = annotations["metadata"]["tile_id"]

            for annotation in annotations["annotations"]:
                out["instances"].append(ProcessedInstance.from_coco_dict(annotation))
        return out
