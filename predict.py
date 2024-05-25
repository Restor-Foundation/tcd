import argparse
import os

from hydra import compose, initialize

from tcd_pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["instance", "semantic"])
    parser.add_argument("input", type=str, help="Path to input image (i.e. GeoTIFF)")
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

    runner = Pipeline(cfg)
    res = runner.predict(args.input)

    res.serialise(args.output)
    res.save_masks(args.output)

    if cfg.model.task == "instance_segmentation":
        res.save_shapefile(os.path.join(args.output, "instances.shp"))
        res.visualise(output_path=os.path.join(args.output, "tree_predictions.jpg"))


if __name__ == "__main__":
    main()
