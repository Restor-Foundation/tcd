import argparse
import logging
import os
import time
from pathlib import Path

import fiona
import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.warp import transform_geom
from shapely.geometry import box, mapping, shape

from tcd_pipeline.modelrunner import ModelRunner
from tcd_pipeline.report import generate_report
from tcd_pipeline.util import convert_to_projected

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_shape_from_geofile(geometry_path):
    geometries = []
    features = []
    with fiona.open(geometry_path, "r") as geojson:
        for feature in geojson:
            geometry = feature["geometry"]
            geometries.append(geometry)
            features.append(feature)
        geom_crs = geojson.crs

    return geometries, geom_crs


def get_intersecting_geometries(geometries, geom_crs, raster_path):
    intersecting_geometries = []

    with rasterio.open(raster_path) as raster:
        raster_bound = box(*raster.bounds)

        idx = 0
        for j, geometry in enumerate(geometries):
            transformed_geometry = transform_geom(geom_crs, raster.crs, geometry)

            if shape(transformed_geometry).intersects(raster_bound):
                intersecting_geometries.append(transformed_geometry)
                idx += 1

    return intersecting_geometries


def predict(image, config, tile_size=2048, gsd_m=0.1, warm=False):
    tstart = time.time()

    runner = ModelRunner(config)
    runner.config.data.gsd = gsd_m
    runner.config.data.tile_size = tile_size
    results = runner.predict(image, warm_start=warm)

    tend = time.time()
    results.prediction_time_s = tend - tstart
    return results


def main(args):
    if args.output is None:
        args.output = os.path.join(
            os.path.dirname(args.image), f"{Path(args.image).stem}_pred"
        )

    os.makedirs(args.output, exist_ok=True)

    logger.info(f"Storing output in {args.output}")

    image_path = args.image
    assert os.path.exists(image_path)
    file_name = os.path.basename(image_path)

    if args.resample:
        resampled_image_path = os.path.join(args.output, file_name)

        if not os.path.exists(resampled_image_path):
            logger.info("Resampling")

            convert_to_projected(
                image_path, resampled_image_path, resample=True, target_gsd_m=args.gsd
            )

            assert os.path.exists(image_path)
        else:
            logger.info("Skipping resample")

        image_path = resampled_image_path

    if args.geometry:
        geoms, geom_crs = load_shape_from_geofile(args.geometry)
        intersections = get_intersecting_geometries(geoms, geom_crs, image_path)

        output_paths = []
        geometries = []
        for idx, geom in enumerate(intersections):
            geometries.append(geom)
            output_paths.append(os.path.join(args.output, f"{idx}"))
    else:
        geometries = [None]
        output_paths = [args.output]

    assert len(output_paths) > 0

    out_path = os.path.join(args.output, "semantic_segmentation")
    if args.overwrite and not os.path.exists(os.path.join(out_path, "results.json")):
        result = predict(
            image_path,
            config=args.semantic_seg,
            warm=False,
            tile_size=args.tile_size,
            gsd_m=args.gsd,
        )
        result.serialise(output_folder=out_path)
    else:
        logger.info("Using existing semantic segmentation results")

    if not args.semantic_only:
        out_path = os.path.join(args.output, "instance_segmentation")
        if args.overwrite or not os.path.exists(os.path.join(out_path, "results.json")):
            result = predict(
                image_path,
                config=args.instance_seg,
                warm=False,
                tile_size=args.tile_size,
                gsd_m=args.gsd,
            )
            result.serialise(output_folder=out_path)
        else:
            logger.info("Using existing instance segmentation results")

    from tcd_pipeline.result import InstanceSegmentationResult, SegmentationResult

    for idx, (geom, path) in enumerate(zip(geometries, output_paths)):
        serialise_path = os.path.join(path, "semantic_segmentation")
        os.makedirs(serialise_path, exist_ok=True)

        result_segmentation = SegmentationResult.load_serialisation(
            input_file=os.path.join(
                args.output, "semantic_segmentation", "results.json"
            ),
            image_path=image_path,
        )

        if geom is not None:
            result_segmentation.set_roi(geom)

        result_segmentation.set_threshold(0.5)
        result_segmentation.save_masks(output_path=serialise_path)
        result_segmentation.visualise(
            output_path=serialise_path, dpi=500, max_pixels=(2048, 2048)
        )

        serialise_path = os.path.join(path, "instance_segmentation")
        os.makedirs(serialise_path, exist_ok=True)

        result_instance = None
        if not args.semantic_only:
            result_instance = InstanceSegmentationResult.load_serialisation(
                input_file=os.path.join(
                    args.output, "instance_segmentation", "results.json"
                ),
                image_path=image_path,
            )

            if geom is not None:
                result_instance.set_roi(geom)

            result_instance.set_threshold(0.5)
            result_instance.save_masks(output_path=serialise_path)
            result_instance.visualise(
                output_path=serialise_path, dpi=500, max_pixels=(2048, 2048)
            )

        generate_report(image_path, path, result_instance, result_segmentation, geom)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--image", help="Input image (GeoTIFF orthomosasic)", required=True
    )
    parser.add_argument(
        "-g", "--geometry", help="Input shapefile (GeoJSON or Shapefile)"
    )
    parser.add_argument("-s", "--tile-size", help="Tile size", default=2048)
    parser.add_argument("-o", "--output", help="Working and output directory")
    parser.add_argument("-r", "--resample", help="Resample image", action="store_true")
    parser.add_argument(
        "--semantic_only",
        help="Only perform semantic segmentation",
        action="store_true",
    )
    parser.add_argument(
        "--overwrite",
        help="Overwrite existing results, otherwise use old ones.",
        action="store_true",
    )
    parser.add_argument("--gsd", type=float, default=0.1)
    parser.add_argument(
        "--instance_seg",
        help="Instance segmentation config",
        default="config/detectron_tta.yaml",
    )
    parser.add_argument(
        "--semantic_seg",
        help="Semantic segmentation config",
        default="config/segmentation_tta.yaml",
    )

    args = parser.parse_args()

    main(args)
