import matplotlib.pyplot as plt
import numpy as np
import rasterio

import model

runner = model.ModelRunner("default.yaml")
image_path = "./data/5c15321f63d9810007f8b06f_10_00000.tif"

results = runner.detect_tiled(image_path, tile_size=512, pad=100, skip_empty=True)

image = rasterio.open(image_path)

plt.figure(figsize=(15, 15))

extent = [image.bounds[0], image.bounds[2], image.bounds[1], image.bounds[3]]
plt.imshow(image.read().transpose((1, 2, 0)), extent=extent)
ax = plt.gca()

threshold = 0.5
image_mask = runner.merge_tiled_results(results, image, threshold)

for i, result in enumerate(results):

    _, bbox = result

    rect = plt.Rectangle(
        xy=(bbox.minx, bbox.miny),
        width=bbox.maxx - bbox.minx,
        height=bbox.maxy - bbox.miny,
        alpha=0.25,
        linewidth=4,
        edgecolor="red",
        facecolor="none",
    )

    ax.add_patch(rect)

# Trees
masked = np.ma.masked_where(image_mask[:, :, 0] == 0, image_mask[:, :, 0])
plt.imshow(masked, alpha=0.8, extent=extent, cmap="Blues_r")

# Trees
masked = np.ma.masked_where(image_mask[:, :, 1] == 0, image_mask[:, :, 1])
plt.imshow(masked, alpha=0.8, extent=extent, cmap="Reds_r")

plt.savefig("5c15321f63d9810007f8b06f_10_00000.png")
