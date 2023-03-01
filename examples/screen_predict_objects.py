import cv2
import numpy as np
import rasterio
from affine import Affine
from mss import mss
from PIL import Image

from tcd_pipeline.modelrunner import ModelRunner

"""
This script is a neat demonstration of running a canopy model
live on your current screen contents. This allows you to
experiment with imagery from any source that you can display
without needing to worry about dataloading. Try it with 
Google Maps open! You will have to experiment to find out
what resolution works well.

The mss library is used for screen capture. A crop of the screen
is then sent to the model for prediction and the results and
crop are displayed.

If you don't have a  GPU then the feedback loop will be very slow
for large images but a modern laptop should be able to handle 
256 x 256 fairly fast.
"""

# Base segmentation model

runner = ModelRunner("config/base_detectron.yaml")

# Adjust this to whatever portion of your screen you want to capture.
mon = {"left": 1024, "top": 256, "width": 1024, "height": 1024}

# Load a dummy image to fake the transform and CRS
# because we aren't predicting on georeferenced data
with rasterio.open("data/5c15321f63d9810007f8b06f_10_00000.tif") as src:
    dummy_transform = src.transform
    dummy_crs = src.crs

with mss() as sct:
    while True:
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
            results = runner.predict(dataset, warm_start=False)

        # thresh_mask = results.confidence_map < 0.25
        # results.confidence_map[thresh_mask] = 0

        # Colour map the confidence mask of the image
        mask = (results.tree_mask * 255.0).astype(np.uint8)
        mask_coloured = cv2.applyColorMap(mask, cv2.COLORMAP_INFERNO)

        # Stack with the RGB for visualisation
        alpha = 0.5
        beta = 1.0 - alpha
        combine = cv2.addWeighted(
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR), alpha, mask_coloured, beta, gamma=0
        )

        cv2.imshow("Canopy predictions", combine)
        if cv2.waitKey(33) & 0xFF in (
            ord("q"),
            27,
        ):
            break
