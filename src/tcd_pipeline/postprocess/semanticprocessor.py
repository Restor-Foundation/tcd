import logging
import os
from glob import glob
from typing import Any, Optional

import numpy as np
import rasterio
from shapely.geometry import box

from tcd_pipeline.cache import GeotiffSemanticCache, PickleSemanticCache
from tcd_pipeline.result.semanticsegmentationresult import (
    SemanticSegmentationResult,
    SemanticSegmentationResultFromGeotiff,
)

from .postprocessor import PostProcessor

logger = logging.getLogger(__name__)


class SemanticSegmentationPostProcessor(PostProcessor):
    def __init__(self, config: dict, image: Optional[rasterio.DatasetReader] = None):
        """Initializes the PostProcessor.

        Accepts an optional image, otherwise it's expected this is initialised when cache is
        used.

        Args:
            config (DotMap): the configuration
            image (DatasetReader): input rasterio image
        """
        super().__init__(config, image)
        self.cache_suffix = "segmentation"

    def setup_cache(self):
        cache_format = self.config.postprocess.cache_format

        if cache_format == "pickle":
            self.cache = PickleSemanticCache(
                self.cache_folder,
                self.image.name,
                self.config.data.classes,
                self.cache_suffix,
            )
        elif cache_format == "geotiff":
            self.cache = GeotiffSemanticCache(
                cache_folder=self.cache_folder,
                image_path=self.image.name,
                classes=self.config.data.classes,
                cache_suffix=self.cache_suffix,
            )
        else:
            raise NotImplementedError(f"Cache format {cache_format} is unsupported")

    def cache_result(self, result: dict) -> None:
        """Cache a single tile result

        Args:
            result (dict): result containing the confidence mask (key = mask) and the bounding box (key = bbox)

        """
        if isinstance(self.cache, GeotiffSemanticCache):
            minx, miny, maxx, maxy = result["bbox"].bounds
            pred = (255 * result["predictions"].cpu().numpy()).astype(np.uint8)

            # TODO: Check the validity of this for smaller images where the tile
            # overlap might be different.
            pad = int(self.config.data.tile_overlap / 2)

            pad_left = pad if minx != 0 else 0
            pad_right = pad if maxx != self.image.width else 1
            pad_top = pad if maxy != self.image.height else 1
            pad_bottom = pad if miny != 0 else 0

            pred = pred[:, pad_bottom:-pad_top, pad_left:-pad_right]
            inset_box = box(
                minx + pad_left, miny + pad_bottom, maxx - pad_right, maxy - pad_top
            )

            # Since we crop the bounding box of the tile to the interior region
            # for merging, we only store this inner bbox and not the outer
            self.cache.save(pred, inset_box)
        else:
            self.cache.save(result["predictions"].cpu().numpy(), result["bbox"])

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

        logger.debug("Collecting results")
        assert self.image is not None

        if isinstance(self.cache, GeotiffSemanticCache):
            import shutil

            logger.info("Compressing cache tiles")
            self.cache.compress_tiles()
            shutil.copytree(
                self.cache.cache_folder, self.config.data.output, dirs_exist_ok=True
            )
            output_files = glob(
                os.path.join(
                    os.path.abspath(self.config.data.output), "*_segmentation.tif"
                )
            )
            assert (
                len(output_files) > 0
            ), "Couldn't find any output tiles (looking for *_segmentation.tif)"

            self.cache.generate_vrt(
                files=output_files, root=os.path.abspath(self.config.data.output)
            )

            logger.info("Prediction complete, returning result")
            return SemanticSegmentationResultFromGeotiff(
                image=self.image,
                prediction=rasterio.open(
                    os.path.join(self.config.data.output, "overview.vrt")
                ),
                config=self.config,
            )
        else:
            if self.config.postprocess.stateful:
                self.cache.load()
                self.results = self.cache.results

            return SemanticSegmentationResult(
                image=self.image,
                tiled_masks=[r["mask"] for r in self.results],
                bboxes=[r["bbox"] for r in self.results],
                config=self.config,
                merge_pad=self.config.postprocess.segmentation_merge_pad,
            )
