# Script to batch convert geotiffs to jpeg or png

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rich.progress import track


def main(args):

    image_paths = glob.glob(os.path.join(args.input, "*.tif"))
    image_paths.extend(glob.glob(os.path.join(args.input, "*.TIF")))
    image_paths.extend(glob.glob(os.path.join(args.input, "*.TIFF")))
    image_paths.extend(glob.glob(os.path.join(args.input, "*.tiff")))

    os.makedirs(args.output, exist_ok=True)

    for image_path in track(image_paths):
        path = Path(image_path)
        with rasterio.open(path) as src:

            img_array = src.read()

            # Gray to RGB
            if src.count == 1:
                # [1,1024,1024] > [3, 1024,1024]
                img_array = np.repeat(img_array, 3, axis=0)

            if src.count > 3:
                img_array = img_array[:3]

            img_array = img_array.transpose((1, 2, 0))

            img = Image.fromarray(img_array)
            img.save(os.path.join(args.output, path.stem + args.ext))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", help="Output folder", default="./")
    parser.add_argument("--ext", help="Output extension", default=".jpg")
    parser.add_argument("input", help="Folder of images to convert")
    args = parser.parse_args()

    main(args)
