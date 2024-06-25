import logging
import os
from abc import abstractmethod
from typing import Any, List, Optional, Union

import rasterio

from tcd_pipeline.cache.cache import ResultsCache
from tcd_pipeline.result.processedresult import ProcessedResult

logger = logging.getLogger(__name__)


class PostProcessor:
    """Processes results from a model, provides support for caching model results
    and keeping track of tile results in the context of the "source" image
    """

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
        self.cache_suffix = None
        self.cache: ResultsCache = None

        if image is not None:
            self.initialise(image)

    @abstractmethod
    def setup_cache(self):
        """
        Initialise the cache. Abstract method, depends on the type of postprocessor (instance, semantic, etc)
        """
        raise NotImplementedError

    def initialise(self, image, warm_start=False) -> None:
        """Initialise the processor for a new image and creates cache
        folders if required.

        Args:
            image (DatasetReader): input rasterio image
            warm_start (bool, option): Whether or not to continue from where one left off. Defaults to False
                                        to avoid unexpected behaviour.
        """
        self.results = []
        self.image = image
        self.tile_count = 0

        # Break early if we aren't doing stateful (cached) post-processing
        if not self.config.postprocess.stateful:
            return

        self.warm_start = warm_start
        self.cache_folder = os.path.abspath(
            os.path.join(
                self.cache_root,
                os.path.splitext(os.path.basename(self.image.name))[0] + "_cache",
            )
        )

        self.setup_cache()

        # Always clear the cache directory if we're doing a cold start
        if warm_start:
            logger.info(f"Attempting to use cached result from {self.cache_folder}")
            # Check to see if we have a bounding box file
            # this stores how many tiles we've processed

            if self.image is not None and os.path.exists(self.cache_folder):
                self.cache.load()

                # We should probably have a strict mode that will error out
                # if there's a cache mismatch
                self.tile_count = len(self.cache)

                if self.tile_count > 0:
                    logger.info(f"Starting from tile {self.tile_count + 1}.")
                    return

        # Otherwise we should clear the cache
        logger.debug(f"Attempting to clear existing cache")
        self.cache.clear()
        self.cache.initialise()

    def add(self, results: List[dict]):
        """
        Add results to the post processor
        """
        for result in results:
            self.tile_count += 1

            # We always want to keep a list of bounding boxes
            new_result = {"tile_id": self.tile_count, "bbox": result["bbox"]}

            # Either cache results, or add to in-memory list
            if self.config.postprocess.stateful:
                logger.debug(f"Saving cache for tile {self.tile_count}")
                self.cache_result(result)

                if self.config.postprocess.debug_images:
                    self.cache.cache_image(self.image, result["window"])

            else:
                new_result |= self._transform(result)

            self.results.append(new_result)

    @abstractmethod
    def merge(self):
        """
        Merge results from overlapping tiles
        """
        raise NotImplementedError

    @abstractmethod
    def process(self) -> ProcessedResult:
        """
        Processes the stored results into a ProcessedResult object that represents
        the complete prediction over the tiled input
        """
        raise NotImplementedError

    @abstractmethod
    def cache_result(self) -> None:
        """
        Store a prediction in the cache
        """
        raise NotImplementedError
