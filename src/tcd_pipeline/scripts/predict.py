import argparse
import logging
import os
from glob import glob

from tcd_pipeline import Pipeline
from tcd_pipeline.pipeline import known_models
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
    parser.add_argument("--only-predict", action="store_true")
    parser.add_argument(
        "--filter", type=str, help="Semantic mask to filter instances, if available"
    )
    parser.add_argument(
        "output",
        type=str,
        help="Path to output folder for predictions, will be created if it doesn't exist",
    )
    parser.add_argument(
        "options",
        nargs=argparse.REMAINDER,
        help="Configuration options to pass to the pipeline, formatted as <key>=<value> with spaces between options.",
    )

    args = parser.parse_args()

    pipeline = Pipeline(args.model, args.options)

    pipeline.config.data.output = args.output
    logger.info(f"Saving results to {pipeline.config.data.output}")

    import rasterio

    image_resolution = rasterio.open(args.input).res[0]
    if image_resolution > 1:
        raise ValueError(f"Image resolution is likely too low, at {image_resolution}")

    os.makedirs(args.output, exist_ok=True)

    try:
        symlink_image = os.path.join(args.output, os.path.basename(args.input))
        if not os.path.exists(symlink_image):
            os.symlink(os.path.abspath(args.input), symlink_image)
    except:
        pass

    # Attempt to resample the image
    if (image_resolution - 0.1) > 1e-4:
        import subprocess
        from pathlib import Path

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
    res = pipeline.predict(input)

    # TODO FIX RESULTS

    if not args.only_predict:
        res.save_masks(args.output)

        if pipeline.config.model.task == "instance_segmentation":
            res.visualise(output_path=os.path.join(args.output, "tree_predictions.jpg"))

    if args.filter and pipeline.config.model.task == "instance_segmentation":
        for shapefile in glob(os.path.join(args.output, "*.shp")):
            if "filter" not in shapefile:
                filter_shapefile(shapefile, args.filter)


if __name__ == "__main__":
    main()