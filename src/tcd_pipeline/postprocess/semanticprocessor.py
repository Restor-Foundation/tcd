import logging
from typing import Any, Optional

import rasterio

from tcd_pipeline.cache.semanticcache import NumpySemanticCache, PickleSemanticCache
from tcd_pipeline.postprocess.postprocessor import PostProcessor
from tcd_pipeline.result.semanticsegmentationresult import SemanticSegmentationResult

logger = logging.getLogger(__name__)


class SemanticSegmentationPostProcessor(PostProcessor):
    def __init__(self, config: dict, image: Optional[rasterio.DatasetReader] = None):
        """Initializes the PostProcessor.

        Accepts an optional image, otherwise it's expected this is initialised when cache is
        used.

        Args:
            config (DotMap): the configuration
            image (DatasetReaer): input rasterio image
        """
        super().__init__(config, image)
        self.cache_suffix = "segmentation"

    def setup_cache(self):
        cache_format = self.config.postprocess.cache_format

        if cache_format == "numpy":
            self.cache = NumpySemanticCache(
                self.cache_folder,
                self.image.name,
                self.config.data.classes,
                self.cache_suffix,
            )
        elif cache_format == "pickle":
            self.cache = PickleSemanticCache(
                self.cache_folder,
                self.image.name,
                self.config.data.classes,
                self.cache_suffix,
            )
        else:
            raise NotImplementedError(f"Cache format {cache_format} is unsupported")

    def cache_result(self, result: dict) -> None:
        """Cache a single tile result

        Args:
            result (tuple[Instance, Bbox]): result containing the confidence mask and the bounding box

        """

        self.cache.save(result["predictions"], result["bbox"])

        if self.config.postprocess.debug_images:
            self.cache_tile_image(result["bbox"])

    def _transform(self, result: dict) -> dict:
        out = {}
        out["mask"] = result["predictions"]

        return out

    def process(self) -> SemanticSegmentationResult:
        """Processes stored and/or cached results

        Returns:
            SegmentationResult: SegmentationResult of the segmentation task
        """

        assert self.image is not None

        if self.config.postprocess.stateful:
            self.cache.load()
            self.results = self.cache.results

        logger.debug("Collecting results")

        return SemanticSegmentationResult(
            image=self.image,
            tiled_masks=[r["mask"] for r in self.results],
            bboxes=[r["bbox"] for r in self.results],
            config=self.config,
            merge_pad=self.config.postprocess.segmentation_merge_pad,
        )
