import argparse
import logging

from tcd_pipeline.util import convert_to_projected

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    """This script resizes an image to a given ground sample distance (GSD) and
    compresses it a JPEG formatted image.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str, help="Input image")
    parser.add_argument(
        "--compress_only", help="Only compress, don't resample", action="store_false"
    )
    parser.add_argument("--gsd", type=float, help="Ground sample distance", default=0.1)
    parser.add_argument("--inplace", help="Operate in place", action="store_true")

    args = parser.parse_args()

    convert_to_projected(
        args.input,
        inplace=(args.inplace == True),
        resample=args.compress_only,
        target_gsd_m=args.gsd,
    )
