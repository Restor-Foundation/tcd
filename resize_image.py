import argparse
import logging

from util import convert_to_projected

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, help="Input image", required=True)
    parser.add_argument("--gsd", type=float, help="Ground sample distance", default=0.1)

    args = parser.parse_args()

    convert_to_projected(
        args.input, inplace=False, resample=True, target_gsd_m=args.gsd
    )
