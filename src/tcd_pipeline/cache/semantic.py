import glob
import logging
import os
import pickle
from typing import List

import numpy as np
from shapely.geometry import box

from .cache import ResultsCache

logger = logging.getLogger(__name__)


class SemanticSegmentationCache(ResultsCache):
    @property
    def results(self) -> List[dict]:
        """
        A list of ProcessedInstances that were stored in the cache folder
        """
        return self._results


class PickleSemanticCache(SemanticSegmentationCache):
    def _find_cache_files(self) -> List[str]:
        return glob.glob(os.path.join(self.cache_folder, f"*_{self.cache_suffix}.pkl"))

    def save(self, mask, bbox: box):
        output = {"mask": mask, "bbox": bbox, "image": self.image_path}

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


class NumpySemanticCache(SemanticSegmentationCache):
    def _find_cache_files(self) -> List[str]:
        return glob.glob(os.path.join(self.cache_folder, f"*_{self.cache_suffix}.npz"))

    def save(self, mask, bbox: box):
        """
        Save cached results in Numpy format.

        Args:
            instances (List[ProcessedInstance]: a list of instances to cache
            bbox: tile bounding box in global image coordinates
        """
        file_name = f"{self.tile_count}_{self.cache_suffix}.npz"
        output_path = os.path.join(self.cache_folder, file_name)

        np.savez_compressed(
            output_path,
            mask=mask,
            bbox=bbox.bounds,
            image=self.image_path,
            tile_count=self.tile_count,
        )

        self.tile_count += 1

    def _load_file(self, cache_file: str) -> dict:
        """Load cached results from MS-COCO format

        Args:
            cache_file (str): Cache filename

        Returns:
            list[ProcessedInstance]: instances

        """

        res = np.load(cache_file)

        out = {}
        out["bbox"] = box(*res["bbox"])
        out["mask"] = res["mask"]
        out["image"] = str(res["image"])
        out["tile_id"] = self.tile_count

        return out
