import argparse
import logging
import os
import time
from pathlib import Path

from tcd_pipeline.modelrunner import ModelRunner
from tcd_pipeline.report import generate_report
from tcd_pipeline.util import convert_to_projected

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def predict_and_serialise(image, config, serialise_path, gsd_m=0.1, warm=False):
    tstart = time.time()
    runner = ModelRunner(config)
    runner.config.data.gsd = gsd_m
    results = runner.predict(image, warm_start=warm)
    tend = time.time()
    results.prediction_time_s = tend - tstart
    results.serialise(output_folder=serialise_path)


def main(args):

    os.makedirs(args.output, exist_ok=True)

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

    # if args.skip_predict:
    predict_and_serialise(
        image_path,
        args.semantic_seg,
        os.path.join(args.output, "semantic_segmentation"),
        warm=False,
        gsd_m=args.gsd,
    )
    predict_and_serialise(
        image_path,
        args.instance_seg,
        os.path.join(args.output, "instance_segmentation"),
        warm=False,
        gsd_m=args.gsd,
    )

    generate_report(image_path, args.output)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--image", help="Input image", required=True)
    parser.add_argument("-o", "--output", help="Working directory", required=True)
    parser.add_argument("-r", "--resample", help="Resample image", action="store_true")
    # parser.add_argument("--skip_predict", help="Resample image", action="store_false")
    parser.add_argument("--gsd", type=float, default=0.1)
    parser.add_argument(
        "--instance_seg", help="Resample image", default="config/detectron_tta.yaml"
    )
    parser.add_argument(
        "--semantic_seg", help="Resample image", default="config/segmentation_tta.yaml"
    )

    args = parser.parse_args()

    main(args)
