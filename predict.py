import argparse
import logging
import os
from glob import glob

from hydra import compose, initialize

from tcd_pipeline import Pipeline
from tcd_pipeline.util import filter_shapefile

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["instance", "semantic"])
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

    initialize(version_base=None, config_path="src/tcd_pipeline/config")
    overrides = args.options if args.options != [] else None
    cfg = compose(config_name=args.mode, overrides=overrides)

    cfg.data.output = args.output
    logger.info(f"Saving results to {cfg.data.output}")

    runner = Pipeline(cfg)
    res = runner.predict(args.input)

    if not args.only_predict:
        res.serialise(args.output)
        res.save_masks(args.output)

        if cfg.model.task == "instance_segmentation":
            res.visualise(output_path=os.path.join(args.output, "tree_predictions.jpg"))

    if args.filter and cfg.model.task == "instance_segmentation":
        for shapefile in glob(os.path.join(args.output, "*.shp")):
            if "filter" not in shapefile:
                filter_shapefile(shapefile, args.filter)


if __name__ == "__main__":
    main()
