import argparse
import logging
import os
from typing import Optional, Union

import fiona
import matplotlib.pyplot as plt
import rasterio
import rasterio.windows
import shapely
import torch
from rasterio.enums import Resampling
from shapely.geometry import box
from torchmetrics import (
    Accuracy,
    Dice,
    F1Score,
    JaccardIndex,
    MetricCollection,
    Precision,
    Recall,
)
from tqdm import tqdm

from tcd_pipeline.data.tiling import generate_tiles

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_overlap(ras1, ras2):
    x_scale = ras2.transform.a / ras1.transform.a
    y_scale = ras2.transform.e / ras1.transform.e

    # scale image transform
    transform = ras2.transform * ras2.transform.scale(
        (ras2.width / ras2.shape[-1]), (ras2.height / ras2.shape[-2])
    )

    ext1 = box(*ras1.bounds)
    ext2 = box(*ras2.bounds)
    intersection = ext1.intersection(ext2)
    window = rasterio.windows.from_bounds(*intersection.bounds, ras1.transform)

    return window, transform, x_scale, y_scale


def sample_raster(src, bounds, x_scale, y_scale, transform):
    win2 = rasterio.windows.from_bounds(*bounds, transform)

    data = src.read(
        window=win2,
        out_shape=(src.count, int(win2.height * y_scale), int(win2.width * x_scale)),
        resampling=Resampling.bilinear,
    )

    return data


def maybe_warp_geometry(
    image,
    shape: Union[dict, shapely.geometry.Polygon],
    crs: Optional[rasterio.crs.CRS] = None,
):
    """Filter by geometry, should be a simple Polygon i.e. a
    convex hull that defines the region of interest for analysis.

    Args:
        shape (dict): shape to filter the data
        crs: Coordinate reference system for the region, usually
                assumed to be the CRS of the image

    """

    if crs is not None and crs != image.crs:
        logger.warning("Geometry CRS is not the same as the image CRS, warping")
        shape = rasterio.warp.transform_geom(crs, image.crs, shape)

    if not isinstance(shape, shapely.geometry.Polygon):
        shape = shapely.geometry.shape(shape)

    if isinstance(shape, shapely.geometry.MultiPolygon):
        shape = shape.geoms[0]

    if not isinstance(shape, shapely.geometry.Polygon):
        logger.warning("Input shape is not a polygon, not applying filter")
        return

    return shapely.geometry.polygon.orient(shape)


def load_shape_from_geofile(geometry_path):
    geometries = []
    features = []
    with fiona.open(
        geometry_path, "r", enabled_drivers=["GeoJSON", "ESRI Shapefile"]
    ) as geo:
        for feature in geo:
            geometry = feature["geometry"]
            geometries.append(geometry)
            features.append(feature)
        geom_crs = geo.crs

    return geometries[0], geom_crs


def evaluate_semantic(args, threshold=1):
    """
    Evaluate a semantic segmentation prediction against a raster ground truth.
    """
    metrics = MetricCollection(
        {
            "accuracy": Accuracy(task="binary"),
            "iou": JaccardIndex(task="binary"),
            "f1": F1Score(task="binary"),
            "precision": Precision(task="binary"),
            "recall": Recall(task="binary"),
            "dice": Dice(),
        }
    )

    with rasterio.open(args.prediction) as pred:
        geometry, geometry_crs = load_shape_from_geofile(args.geometry)

        valid_region = maybe_warp_geometry(pred, geometry, geometry_crs)

        tile_size = args.tile_size

        with rasterio.open(args.ground_truth) as gt:
            # Get extent of bounding region
            overlap_window, transform, x_scale, y_scale = get_overlap(pred, gt)

            # Generate tiles within the region
            tiles = generate_tiles(
                overlap_window.height, overlap_window.width, tile_size
            )

            pbar = tqdm(enumerate(tiles), total=len(tiles))
            for idx, tile in pbar:
                # Generate a window for the current tile
                minx, miny, maxx, maxy = tile.bounds
                window = rasterio.windows.Window(
                    minx + overlap_window.col_off,
                    miny + overlap_window.row_off,
                    width=int(maxx - minx),
                    height=int(maxy - miny),
                )

                pred_data = pred.read(window=window)[0] / 255.0
                tile_bounds = rasterio.windows.bounds(window, pred.transform)

                # Sample the tile from the other raster, and scale if necessary
                gt_data = sample_raster(gt, tile_bounds, x_scale, y_scale, transform)[0]

                # Get intersection between the current tile and the
                # mask geometry
                intersection = valid_region.intersection(box(*tile_bounds))
                if shapely.is_empty(intersection):
                    continue

                # Generate a mask for the current tile, invert so that we can
                # get a mask of valid pixels. NB shape is a numpy shape, so
                # (height, width) order
                mask = rasterio.features.geometry_mask(
                    [intersection],
                    out_shape=(window.height, window.width),
                    transform=rasterio.windows.transform(window, pred.transform),
                    invert=True,
                )

                # print(pred_data.shape, gt_data.shape, mask.shape)
                res = metrics(
                    torch.from_numpy(pred_data[mask]),
                    torch.from_numpy(gt_data[mask] > threshold),
                )
                pbar.set_postfix_str(res)
                pbar.update()

                # Debug
                """
                fig = plt.figure()
                plt.subplot(131)
                plt.imshow(pred_data * mask > 0.5)
                plt.subplot(132)
                plt.imshow(gt_data > 1)
                plt.subplot(133)
                plt.imshow(mask)
                plt.title(res)
                plt.savefig(os.path.join("temp", f"out_tiles_{idx}.jpg"), bbox_inches='tight')
                plt.close(fig)
                """

            # Summarise
            with open(args.result, "w") as fp:
                import json

                json.dump({k: float(v) for k, v in metrics.compute().items()}, fp)


def evaluate_instance(args):
    from pycocotools import cocoeval
    from pycocotools.coco import COCO

    gt = COCO(args.ground_truth)
    pred = gt.loadRes(args.prediction)

    eval = cocoeval.COCOeval(gt, pred, iouType="segm")
    eval.evaluate()
    eval.accumulate()
    eval.summarize()

    with open(args.result, "w") as fp:
        fp.write(eval)


def main(args):
    if args.task == "semantic":
        evaluate_semantic(args)
    elif args.task == "instance":
        evaluate_instance(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", help="Root folder", choices=["semantic", "instance"])
    parser.add_argument("prediction", help="Prediction result (GeoTIFF or COCO JSON)")
    parser.add_argument("ground_truth", help="Ground truth (GeoTIFF or COCO JSON)")
    parser.add_argument("result", help="Output metric file")
    parser.add_argument(
        "-g", "--geometry", help="Region of interest to perform analysis on"
    )
    parser.add_argument(
        "-s",
        "--tile_size",
        default=10000,
        help="Tile size used for metric evaluation - RAM limited",
    )
    args = parser.parse_args()

    main(args)
