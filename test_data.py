import rasterio as rio

from data import dataloader_from_image

image = rio.open("./data/5f058f16ce2c9900068d83ed.tif")
tile_size_px = 1024
stride_px = 256

dataloader = dataloader_from_image(
    image, tile_size_px, stride_px, gsd_m=0.1, batch_size=1
)

print(image.res)

for batch in dataloader:
    image = batch["image"][0].float()

    print(image.shape, tile_size_px)

    assert image.shape[0] == tile_size_px
