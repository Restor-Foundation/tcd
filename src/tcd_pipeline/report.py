import json
import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from subprocess import check_output

import fiona
import matplotlib.pyplot as plt
import natsort
import numpy as np
import rasterio
import rasterio.features
import rasterio.plot
import shapely.geometry
from affine import Affine
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds

from tcd_pipeline.result import InstanceSegmentationResult, SegmentationResult

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def generate_report(image, output_root):
    report_folder = os.path.join(output_root, "report")

    tstart = time.time()
    logging.info("Loading instance segmentation result")
    results_instances = InstanceSegmentationResult.load_serialisation(
        input_file=os.path.join(output_root, "instance_segmentation", "results.json"),
        image_path=image,
    )
    save_instance_segmentation(results_instances, report_folder)

    logging.info("Loading semantic segmentation result")
    results_segmentation = SegmentationResult.load_serialisation(
        os.path.join(output_root, "semantic_segmentation", "results.json"),
        image_path=image,
    )

    save_segmentation(results_segmentation, report_folder, geometry=None)

    tend = time.time()

    logging.info("Generating report")
    results_to_report(
        results_instances,
        results_segmentation,
        report_time_s=tend - tstart,
        report_folder=report_folder,
    )


def render_report(output_path, report_results):
    from liquid import Environment, FileSystemLoader, Mode, StrictUndefined

    env = Environment(
        tolerance=Mode.STRICT,
        undefined=StrictUndefined,
        loader=FileSystemLoader("templates/"),
    )

    template = env.get_template("canopy_coverage.html")
    result = template.render(data=report_results, page_title="Canopy coverage report")

    with open(output_path, "w") as fp:
        fp.write(result)

    output_base, _ = os.path.splitext(output_path)

    # Disable JS so we don't mess up with the leaflet page
    check_output(
        [
            "wkhtmltopdf",
            "--enable-local-file-access",
            "--disable-javascript",
            output_path,
            output_base + ".pdf",
        ]
    )

    # Copy needed JS files (leaflet, proj4, proj4leaflet)
    shutil.copy("templates/assets/proj4.js", os.path.dirname(output_path))
    shutil.copy("templates/assets/proj4leaflet.js", os.path.dirname(output_path))


def bundle_geojson(src):
    import json

    import shapefile

    with shapefile.Reader(src) as shp:
        geojson_data = shp.__geo_interface__

        with open(os.path.join(os.path.dirname(src), "tree_geojson.js"), "w") as fp:

            fp.write("var tree_shapes = ")
            json.dump(geojson_data, fp)


def save_segmentation(results, report_folder, geometry=None):

    file_stem = Path(results.image.name).stem

    if geometry is not None:
        results.filter_geometry(geometry)

    results.set_threshold(0.5)
    results.save_masks(output_path=report_folder)
    results.visualise(output_path=report_folder, dpi=500, max_pixels=(2048, 2048))


def save_instance_segmentation(results, report_folder):
    file_stem = Path(results.image.name).stem
    filename = f"instances_{file_stem}.json"

    threshold = 0.5
    results.set_threshold(threshold)
    results.save_masks(report_folder, prefix="instance_", suffix=f"_{str(threshold)}")
    results.visualise(
        output_path=os.path.join(report_folder, "tcd_overlay.jpg"),
        dpi=500,
        max_pixels=(2048, 2048),
    )

    shapefile_name = f"{file_stem}_tcd_{threshold}.shp"
    shapefile_path = os.path.join(report_folder, shapefile_name)
    results.save_shapefile(shapefile_path)

    bundle_geojson(shapefile_path)

    areas = [
        tree.polygon.area * results.image.res[0] * results.image.res[0]
        for tree in results.get_trees()
    ]

    plt.figure()
    _ = plt.hist(areas, bins=75, range=(0.5, np.quantile(areas, 0.9)))
    plt.xlabel("Area (m$^2$)")
    plt.ylabel("Count")
    plt.savefig(
        fname=os.path.join(report_folder, "area_histogram.jpg"), bbox_inches="tight"
    )


def results_to_report(
    results_instances, results_segmentation, report_time_s, report_folder
):

    os.makedirs(report_folder, exist_ok=True)
    src = results_segmentation.image
    res = src.res[0]

    latlon_bounds = transform_bounds(src.crs, "EPSG:4236", *src.bounds)
    # left, bottom, right, top i.e.

    report_data = {
        "timestamp": datetime.now(),
        "shapefile": False,
        "raw_image": "raw_image.jpg",
        "image_name": os.path.basename(src.name),
        "image_area_m": results_segmentation.num_valid_pixels * (res * res),
        "image_res": res,
        "image_crs": src.crs,
        "map_centre": {
            "lon": (latlon_bounds[2] + latlon_bounds[0]) / 2,
            "lat": (latlon_bounds[3] + latlon_bounds[1]) / 2,
        },
        "map_bounds": {
            "lon": (latlon_bounds[0], latlon_bounds[2]),
            "lat": (latlon_bounds[1], latlon_bounds[3]),
        },
    }

    report_data["segmentation"] = {
        "canopy_mask": "canopy_mask.jpg",
        "canopy_overlay": "canopy_overlay.jpg",
        "confidence_threshold": results_segmentation.confidence_threshold,
        "local_maximum": "local_maximum.jpg",
        "canopy_cover": results_segmentation.canopy_cover,
        "config": results_segmentation.config,
        "model_config": results_segmentation.config["model"]["config"],
    }

    report_data["instance_segmentation"] = {
        "tcd_overlay": "tcd_overlay.jpg",
        "area_histogram": "area_histogram.jpg",
        "geojson": "tree_geojson.js",
        "number_trees": len(results_instances.get_trees()),
        "config": results_instances.config,
        "model_config": results_instances.config["model"]["config"],
    }

    file_stem = Path(src.name).stem

    report_data["valid_pixels"] = results_instances.num_valid_pixels
    report_data["process_time_s"] = report_time_s
    report_data["process_time_seg"] = results_segmentation.prediction_time_s
    report_data["process_time_inst_seg"] = results_instances.prediction_time_s

    render_report(os.path.join(report_folder, f"{file_stem}_report.html"), report_data)
