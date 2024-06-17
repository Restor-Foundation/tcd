import argparse
import time

import cv2
import numpy as np
import rasterio
from mss import mss
from PIL import Image

from tcd_pipeline.pipeline import Pipeline

"""
This script is a neat demonstration of running a canopy model
live on your current screen contents. This allows you to
experiment with imagery from any source that you can display
without needing to worry about dataloading. Try it with 
Google Maps open! You will have to experiment to find out
what resolution/zoom works well.

The mss library is used for screen capture. A crop of the screen
is then sent to the model for prediction and the results and
crop are displayed. You also need opencv-python (not the headless
version) for window support.

If you don't have a  GPU then the feedback loop will be very slow
for large images but a modern laptop should be able to handle 
256 x 256 fairly fast.
"""

# argument parser to switch between models:
parser = argparse.ArgumentParser(
    description="Utility to run tree map prediction based on local screen capture. Hit `q` to exit."
)
parser.add_argument(
    dest="model",
    help="choose model to run",
    type=str,
    choices=["instance", "semantic"],
    default="instance",
)
args = parser.parse_args()

# Base segmentation model
runner = Pipeline(args.model)

# Adjust this to whatever portion of your screen you want to capture.
mon = {"left": 0, "top": 0, "width": 1024, "height": 1024}

# Load a dummy image to fake the transform and CRS
# because we aren't predicting on georeferenced data
with rasterio.open("data/5c15321f63d9810007f8b06f_10_00000.tif") as src:
    dummy_transform = src.transform
    dummy_crs = src.crs


def overlay_two_image(image, overlay, ignore_color=[0, 0, 0]):
    ignore_color = np.asarray(ignore_color)
    mask = ~(overlay == ignore_color).all(-1)
    # Or mask = (overlay!=ignore_color).any(-1)
    out = image.copy()
    out[mask] = image[mask] * 0.6 + overlay[mask] * 0.4

    return out


with mss() as sct:
    win_name = "Canopy predictions"
    try:
        print("Loading window")
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    except cv2.error:
        print(
            "Check that you have opencv-python installed, and not just opencv-python-headless"
        )
        exit(1)

    do_once = True
    canopy_cover = 0

    while True:
        print("Grabbing")
        screenShot = sct.grab(mon)

        img = np.array(
            Image.frombytes(
                "RGB",
                (screenShot.width, screenShot.height),
                screenShot.rgb,
            )
        )

        with rasterio.open(
            "/tmp/new.tif",
            "w+",
            driver="GTiff",
            height=img.shape[0],
            width=img.shape[1],
            count=3,
            dtype=img.dtype,
            crs=dummy_crs,
            transform=dummy_transform,
        ) as dataset:
            for i in range(3):
                dataset.write(img[:, :, i], i + 1)

        # Predict, but don't cache!
        tstart = time.time()
        results = runner.predict("/tmp/new.tif", warm_start=False)
        # if args.model == "semantic":
        #    canopy_cover = results.canopy_cover
        tend = time.time()
        telapsed = tend - tstart

        # thresh_mask = results.confidence_map < 0.25
        # results.confidence_map[thresh_mask] = 0

        # Colour map the confidence mask of the image
        font = cv2.FONT_HERSHEY_DUPLEX
        bottomLeftCornerOfText = (10, 1000)
        fontScale = 1
        fontColor = (255, 255, 255)
        thickness = 1
        lineType = 1

        if args.model == "instance":
            mask = (results.tree_mask * 255.0).astype(np.uint8)

            kernel = np.ones((5, 5), np.uint8)
            mask_erode = cv2.erode(mask, kernel, iterations=1)
            mask -= mask_erode

            # mask_outline = #fe1493
            mask_coloured = np.stack((mask,) * 3, axis=-1) * np.array(
                (147, 20, 254), dtype=mask.dtype
            )

            combine = overlay_two_image(
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR), mask_coloured
            )

            combine = cv2.rectangle(combine, (0, 970), (1024, 1024), (0, 0, 0), -1)
            cv2.putText(
                combine,
                f"Trees detected: {len(results.instances)}, FPS: {1/telapsed:1.2} preds/sec",
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType,
            )
        else:
            mask = results.confidence_map
            mask[mask < 0.25] = 0

            canopy_cover = np.count_nonzero(mask) / ((1024 - 64) * (1024 - 64))

            mask_coloured = (np.stack((mask,) * 3, axis=-1) * 255).astype(np.uint8)
            mask_coloured = cv2.applyColorMap(mask_coloured, cv2.COLORMAP_INFERNO)
            mask_coloured[mask < 0.25] = (0, 0, 0)

            # Stack with the RGB for visualisation
            combine = overlay_two_image(
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR), mask_coloured
            )

            combine = cv2.rectangle(
                combine, (64, 64), (1024 - 64, 1024 - 64), (255, 255, 255), 3
            )
            combine = cv2.rectangle(combine, (0, 970), (1024, 1024), (0, 0, 0), -1)

            cv2.putText(
                combine,
                f"Canopy cover: {canopy_cover:1.2}, FPS: {1/telapsed:1.2} preds/sec",
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType,
            )

        cv2.imshow(win_name, combine)

        if do_once:
            cv2.resizeWindow(win_name, 1024, 1024)
            do_once = False
        if cv2.waitKey(33) & 0xFF in (
            ord("q"),
            27,
        ):
            break
