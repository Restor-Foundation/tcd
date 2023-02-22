import argparse

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import requests


def decode_mask(res, key):
    return np.frombuffer(res[key].encode(), dtype=res["dtype"]).reshape(res["shape"])


def main(args):

    image_path = args.image
    url = f"http://127.0.0.1:8000/predict?mode={args.mode}"
    file = {"file": open(image_path, "rb")}

    res = requests.post(url=url, files=file).json()

    with rasterio.open(image_path) as f:
        array = f.read()
        plt.imshow(array.transpose(1, 2, 0))

        canopy_mask = decode_mask(res["results"], "canopy_mask")
        plt.imshow(np.ma.masked_equal(canopy_mask, False), alpha=0.5, cmap="Greens")

        if args.mode == "instance":
            tree_mask = decode_mask(res["results"], "tree_mask")
            plt.imshow(np.ma.masked_equal(tree_mask, False), alpha=0.5, cmap="Reds")

        plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "image",
        help="Path to image to predict on",
        type=str,
        default="../data/5c15321f63d9810007f8b06f_10_00000.tif",
    )
    parser.add_argument(
        "--mode",
        help="Mode to run in",
        type=str,
        default="semantic",
        choices=["semantic", "instance"],
    )
    parser.add_argument("--tta", help="Use test-time augmentation", action="store_true")
    args = parser.parse_args()

    main(args)
