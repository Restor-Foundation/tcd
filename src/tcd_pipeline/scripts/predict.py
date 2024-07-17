import argparse
import logging
import os
from glob import glob
from pathlib import Path

import rasterio

from tcd_pipeline import Pipeline
from tcd_pipeline.pipeline import known_models
from tcd_pipeline.result.instancesegmentationresult import InstanceSegmentationResult
from tcd_pipeline.util import filter_shapefile

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    model_options = ["instance", "semantic"]
    model_options.extend([k for k in known_models.keys()])

    parser.add_argument(
        "model",
        choices=model_options,
        help=("Model to run.  Allowed values are " + ", ".join(model_options)),
        metavar="model_or_config",
    )
    parser.add_argument("input", type=str, help="Path to input image (i.e. GeoTIFF)")
    parser.add_argument(
        "output",
        type=str,
        help="Path to output folder for predictions, will be created if it doesn't exist",
    )

    parser.add_argument(
        "--resume", help="Attempt to resume prediction", action="store_true"
    )
    parser.add_argument(
        "--filter", type=str, help="Semantic mask to filter instances, if available"
    )
    parser.add_argument(
        "--only-predict",
        type=str,
        help="Don't run any post-prediction tasks like mask generation",
    )

    parser.add_argument(
        "options",
        nargs=argparse.REMAINDER,
        help="Configuration options to pass to the pipeline, formatted as <key>=<value> with spaces between options.",
    )

    args = parser.parse_args()

    pipeline = Pipeline(args.model, args.options)

    pipeline.config.data.output = args.output

    image_resolution = rasterio.open(args.input).res[0]
    if image_resolution > 1:
        raise ValueError(f"Image resolution is likely too low, at {image_resolution}")

    if args.filter:
        assert os.path.exists(args.filter), "Filter map doesn't exist!"

    os.makedirs(args.output, exist_ok=True)

    try:
        symlink_image = os.path.join(args.output, os.path.basename(args.input))
        if not os.path.exists(symlink_image):
            os.symlink(os.path.abspath(args.input), symlink_image)
    except:
        pass

    # Attempt to resample the image
    if (image_resolution - 0.1) > 1e-4:
        resampled_image = os.path.join(args.output, Path(args.input).stem + "_10.vrt")

        if not os.path.exists(resampled_image):
            from tcd_pipeline.util import convert_to_projected

            logger.info("Resampling image to 10 cm/px")
            convert_to_projected(
                args.input, resampled_image, resample=True, target_gsd_m=0.1
            )

        input = resampled_image
    else:
        input = args.input

    # Actually do the prediction
    res = pipeline.predict(input, warm_start=args.resume)
    result_filename = os.path.join(args.output, "instances_processed.shp")

    if args.filter and pipeline.config.model.task == "instance_segmentation":
        logger.info("Filtering")
        result_filename = filter_shapefile(result_filename, args.filter)

    if not args.only_predict:
        if pipeline.config.model.task == "instance_segmentation":
            res = InstanceSegmentationResult.from_shapefile(
                image_path=input, shapefile=result_filename
            )
            res.save_masks(args.output)
            res.visualise(output_path=os.path.join(args.output, "tree_predictions.jpg"))


if __name__ == "__main__":
    main()
