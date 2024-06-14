import json
import logging
import math
import os
import time
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import rasterio
import rasterio.crs
import rasterio.windows
import torch
import yaml
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from shapely.geometry import box
from skimage.transform import resize
from tqdm.auto import tqdm

from tcd_pipeline.util import format_lat_str, format_lon_str

from .processedresult import ProcessedResult

logger = logging.getLogger(__name__)


class SemanticSegmentationResult(ProcessedResult):
    def __init__(
        self,
        image: rasterio.DatasetReader,
        tiled_masks: Optional[list] = [],
        bboxes: list[box] = [],
        confidence_threshold: float = 0.2,
        merge_pad: int = 32,
        config: dict = None,
    ) -> None:
        self.image = image
        self.masks = tiled_masks
        self.bboxes = bboxes
        self.merge_pad = merge_pad
        self.valid_region = None
        self.valid_mask = None
        self.prediction_time_s = -1
        self.config = config

        self.valid_window = rasterio.windows.from_bounds(
            *self.image.bounds, transform=self.image.transform
        )

        self.set_threshold(confidence_threshold)

    def serialise(
        self,
        output_folder: str,
        overwrite: bool = True,
        file_prefix: Optional[str] = "results",
    ) -> None:
        """Serialise raw prediction masks. Masks are stored as NPZ files with the
        keys "mask" and "bbox" as well as a timestamp which can be used as a sanity
        check when loading. A JSON file containing a list of masks will also be created.

        Args:
            output_folder (str): output folder
            overwrite (bool, optional): overwrite existing data, defaults True
            file_prefix (str, optional): file name, defaults to results
        """

        logger.info(f"Serialising results to {output_folder}/{file_prefix}")
        os.makedirs(output_folder, exist_ok=True)

        meta_path = os.path.join(output_folder, f"{file_prefix}.json")

        if os.path.exists(meta_path) and not overwrite:
            logger.error(
                f"Output metadata already exists {meta_path}, will not overwrite."
            )
            return

        timestamp = time.time()
        metadata = {}
        metadata["image"] = self.image.name
        metadata["timestamp"] = int(timestamp)
        metadata["masks"] = []
        metadata["confidence_threshold"] = self.confidence_threshold
        metadata["prediction_time_s"] = self.prediction_time_s
        # metadata["config"] = dict(self.config)
        metadata["hardware"] = self.get_hardware_information()

        for i, item in enumerate(zip(self.masks, self.bboxes)):
            mask, bbox = item
            file_name = f"{file_prefix}_{i}.npz"
            output_path = os.path.join(output_folder, file_name)

            if os.path.exists(output_path) and not overwrite:
                logger.error(
                    f"Output file already exists {output_path}, will not overwrite."
                )
                return
            np.savez_compressed(
                file=output_path,
                mask=mask[0][0],
                image_bbox=mask[0][1],
                bbox=np.array(bbox),
                timestamp=int(timestamp),
            )
            metadata["masks"].append(os.path.abspath(output_path))

        with open(meta_path, "w") as fp:
            json.dump(metadata, fp, indent=1)

    @classmethod
    def load_serialisation(cls, input_file: str, image_path: Optional[str] = None):
        """Loads a ProcessedResult based on a json serialization file.

        Args:
            input_file (str): serialised instance metadata JSON file
            image_path (Optional[str]): image path, optional
        Returns:
            SegmentationResult: SegmentationResult described by the file
        """
        tiled_masks = []
        bboxes = []

        with open(input_file, "r") as fp:
            metadata = json.load(fp)

        image_path = image_path if image_path else metadata["image"]
        image = rasterio.open(image_path)

        for mask_file in metadata["masks"]:
            data = np.load(mask_file, allow_pickle=True)

            tiled_masks.append([[data["mask"], data["image_bbox"]]])
            bboxes.append(box(*data["bbox"]))

            if data["timestamp"] != metadata["timestamp"]:
                logger.error(
                    "Timestamp in mask and metadat file don't match. Corrupted export?"
                )

        res = cls(
            image=image,
            tiled_masks=tiled_masks,
            bboxes=bboxes,
            confidence_threshold=metadata["confidence_threshold"],
            config=metadata["config"],
        )

        res.prediction_time_s = metadata["prediction_time_s"]

        return res

    def set_threshold(self, new_threshold: int) -> None:
        """Sets the threshold of the ProcessedResult, also regenerates
                prediction masks
        ?
                Args:
                    new_threshold (double): new confidence threshold
        """
        self.confidence_threshold = new_threshold
        self._generate_masks()

    def save_masks(
        self,
        output_path: str,
        suffix: Optional[str] = "",
        prefix: Optional[str] = "",
        pad=0,
    ) -> None:
        """Save prediction masks for tree and canopy. If a source image is provided
        then it is used for georeferencing the output masks.

        Args:
            output_path (str): folder to store data
            suffix (str, optional): mask filename suffix
            prefix (str, optional): mask filename prefix

        """

        os.makedirs(output_path, exist_ok=True)

        canopy_mask = np.array(self.mask)

        if pad > 0:
            canopy_mask[:, :pad] = 0
            canopy_mask[:pad, :] = 0
            canopy_mask[:, -pad:] = 0
            canopy_mask[-pad:, :] = 0

        self._save_mask(
            mask=canopy_mask,
            output_path=os.path.join(output_path, f"{prefix}canopy_mask{suffix}.tif"),
        )

        if self.confidence_map.dtype != np.uint8:
            confidence_mask = np.array((255 * self.confidence_map)).astype(np.uint8)
        else:
            confidence_mask = self.confidence_map

        if pad > 0:
            confidence_mask[:, :pad] = 0
            confidence_mask[:pad, :] = 0
            confidence_mask[:, -pad:] = 0
            confidence_mask[-pad:, :] = 0

        self._save_mask(
            mask=confidence_mask,
            output_path=os.path.join(
                output_path, f"{prefix}canopy_confidence{suffix}.tif"
            ),
            binary=False,
        )

    def _generate_masks(self, average=True) -> npt.NDArray:
        """
        Merges segmentation masks following the strategy outlined in:
        https://arxiv.org/ftp/arxiv/papers/1805/1805.12219.pdf

        1) We clip masks by a fixed amount before merging, this limits
        the effect of edge effects on the final mask.

        2) We merge masks by taking the average value at each overlap

        """

        pad = self.merge_pad
        self.mask = np.zeros(self.image.shape, dtype=bool)
        self.confidence_map = np.zeros(self.image.shape)

        p = torch.nn.Softmax2d()

        for mask, bbox in list(zip(self.masks, self.bboxes)):
            confidence = p(torch.Tensor(mask)).numpy()

            """
            # pred = torch.argmax(confidence, dim=0).numpy()
            _, height, width = confidence.shape

            pad_slice = (
                slice(pad, min(height, bbox.height) - pad),
                slice(pad, min(width, bbox.width) - pad),
            )
            """

            from tcd_pipeline.util import paste_array

            minx, miny, _, _ = bbox.bounds

            paste_array(
                dst=self.confidence_map,
                src=confidence[1][pad:-pad, pad:-pad],
                offset=(int(minx) + pad, int(miny) + pad),
            )
            """
            # TODO check appropriate merge strategy
            if average:
                self.confidence_map[bbox.miny : bbox.maxy, bbox.minx : bbox.maxx][
                    pad_slice
                ] = np.maximum(
                    self.confidence_map[bbox.miny : bbox.maxy, bbox.minx : bbox.maxx][
                        pad_slice
                    ],
                    confidence[1][pad_slice],
                )
            else:
                self.confidence_map[bbox.miny : bbox.maxy, bbox.minx : bbox.maxx][
                    pad_slice
                ] = confidence[1][pad_slice]
            """

        self.mask = self.confidence_map > self.confidence_threshold

        if self.valid_mask is not None:
            self.mask = self.mask[self.valid_window.toslices()] * self.valid_mask
            self.confidence_map = (
                self.confidence_map[self.valid_window.toslices()] * self.valid_mask
            )

        return

    def visualise(
        self,
        dpi=400,
        max_pixels: Optional[tuple[int, int]] = None,
        output_path=None,
        color_trees: Optional[tuple[int, int, int]] = (255, 105, 180),
        alpha: Optional[float] = 0.5,
        **kwargs: Any,
    ) -> None:
        """Visualise the results of the segmentation. If output path is not provided, the results
        will be displayed.

        Args:
            dpi (int, optional): dpi of the output image. Defaults to 200.
            max_pixels: maximum image size
            output_path (str, optional): path to save the output plots. Defaults to None.
            color_trees (tuple, optional): RGB tuple defining the colour for tree annotation
            alpha (float, optional): Alpha opacity for confidence mask when overlaid on original image
            **kwargs (Any): remaining arguments passed to figure creation

        """

        confidence_map = self.confidence_map

        reshape_factor = 1
        if max_pixels is not None:
            reshape_factor = min(
                max_pixels[0] / self.valid_window.height,
                max_pixels[1] / self.valid_window.width,
            )
            reshape_factor = min(reshape_factor, 1)

        shape = (
            math.ceil(self.valid_window.height * reshape_factor),
            math.ceil(self.valid_window.width * reshape_factor),
        )

        vis_image = self.image.read(
            out_shape=(self.image.count, shape[0], shape[1]),
            resampling=Resampling.bilinear,
            masked=True,
            window=self.valid_window,
        ).transpose(1, 2, 0)

        if self.valid_mask is not None:
            if reshape_factor != 1:
                vis_image = vis_image * np.expand_dims(
                    resize(self.valid_mask, shape), -1
                )
            else:
                vis_image = vis_image * np.expand_dims(self.valid_mask, -1)

        resized_confidence_map = confidence_map
        if reshape_factor < 1:
            resized_confidence_map = resize(confidence_map, shape)

        # Normal figure
        fig = plt.figure(dpi=dpi, **kwargs)
        ax = plt.axes()
        ax.tick_params(axis="both", which="major", labelsize="x-small")
        ax.tick_params(axis="both", which="minor", labelsize="xx-small")

        latlon_bounds = transform_bounds(
            self.image.crs,
            "EPSG:4236",
            *rasterio.windows.bounds(self.valid_window, self.image.transform),
        )
        ax.imshow(
            vis_image,
            extent=[
                latlon_bounds[0],
                latlon_bounds[2],
                latlon_bounds[1],
                latlon_bounds[3],
            ],
        )

        # ax.set_xticks(ax.get_xticks()[::2])
        # ax.set_yticks(ax.get_yticks()[::2])

        ax.set_xticklabels([format_lon_str(x) for x in ax.get_xticks()], rotation=45)
        ax.set_yticklabels([format_lat_str(y) for y in ax.get_yticks()], rotation=45)

        if output_path is not None:
            plt.savefig(os.path.join(output_path, "raw_image.jpg"), bbox_inches="tight")

        # Canopy Mask
        fig = plt.figure(dpi=dpi, **kwargs)
        ax = plt.axes()
        ax.tick_params(axis="both", which="major", labelsize="x-small")
        ax.tick_params(axis="both", which="minor", labelsize="xx-small")
        ax.imshow(vis_image)

        confidence_mask_image = np.zeros(
            (*resized_confidence_map.shape, 4), dtype=np.uint8
        )
        confidence_mask_image[
            resized_confidence_map > self.confidence_threshold
        ] = list(color_trees) + [255]
        ax.imshow(confidence_mask_image, alpha=alpha)

        if output_path is not None:
            plt.savefig(
                os.path.join(output_path, "canopy_overlay.jpg"), bbox_inches="tight"
            )

        # Confidence Map
        fig = plt.figure(dpi=dpi, **kwargs)
        ax = plt.axes()
        ax.tick_params(axis="both", which="major", labelsize="x-small")
        ax.tick_params(axis="both", which="minor", labelsize="xx-small")
        import matplotlib.colors

        palette = np.array(
            [
                (1, 1, 1, 0),
                (218 / 255, 215 / 255, 205 / 255, 1),
                (163 / 255, 177 / 255, 138 / 255, 1),
                (88 / 255, 129 / 255, 87 / 255, 1),
                (58 / 255, 9 / 255, 64 / 255, 1),
                (52 / 255, 78 / 255, 65 / 255, 1),
            ]
        )

        cmap = matplotlib.colors.ListedColormap(colors=palette)
        bounds = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

        im = ax.imshow(resized_confidence_map, cmap=cmap, norm=norm)
        cax = fig.add_axes(
            [
                ax.get_position().x1 + 0.01,
                ax.get_position().y0,
                0.02,
                ax.get_position().height,
            ]
        )

        cbar = plt.colorbar(
            im,
            cax=cax,
            extend="both",
            ticks=bounds,
            spacing="proportional",
            orientation="vertical",
        )
        cbar.set_label("Confidence", size="x-small")
        cbar.ax.tick_params(labelsize="xx-small")

        if output_path is not None:
            plt.savefig(
                os.path.join(output_path, "canopy_mask.jpg"), bbox_inches="tight"
            )

        if output_path is None:
            plt.show()

    def _filter_roi(self):
        if self.valid_region is not None:
            self.valid_window = rasterio.features.geometry_window(
                self.image, [self.valid_region]
            )

            self.valid_mask = rasterio.features.geometry_mask(
                [self.valid_region],
                out_shape=self.image.shape,
                transform=self.image.transform,
                invert=True,
            )[self.valid_window.toslices()]

        else:
            logger.warning("Unable to filter instances as no ROI has been set.")

    def __str__(self) -> str:
        """String representation, returns canopy cover for image."""
        return (
            f"ProcessedSegmentationResult(n_trees={len(self.get_local_maxima())},"
            f" canopy_cover={self.canopy_cover:.4f})"
        )

    def _repr_html_(self):
        # Save the plot to a SVG buffer
        from io import BytesIO

        buf = BytesIO()
        plt.imshow(self.confidence_map)
        plt.savefig(buf, format="svg")
        plt.tight_layout()
        plt.close()
        buf.seek(0)
        return buf.getvalue().decode("utf-8")


class SemanticSegmentationResultFromGeotiff(SemanticSegmentationResult):
    def __init__(
        self,
        image: rasterio.DatasetReader,
        prediction: rasterio.DatasetReader,
        confidence_threshold: float = 0.2,
        config: dict = None,
    ) -> None:
        self.image = image
        self.confidence_map = prediction.read()[0]
        self.valid_region = None
        self.valid_mask = None
        self.config = config
        self.prediction_time_s = -1

        self.valid_window = rasterio.windows.from_bounds(
            *self.image.bounds, transform=self.image.transform
        )

        self.set_threshold(confidence_threshold)

    def _generate_masks(self):
        self.mask = self.confidence_map > self.confidence_threshold

        if self.valid_mask is not None:
            self.mask = self.mask[self.valid_window.toslices()] * self.valid_mask
            self.confidence_map = (
                self.confidence_map[self.valid_window.toslices()] * self.valid_mask
            )

        return

    def serialise(self, *args, **kwargs):
        pass

    def load_serialisation(self):
        raise NotImplementedError(
            "This is not required for a result based on a GeoTIFF cache"
        )
