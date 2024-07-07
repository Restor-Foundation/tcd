import os
import random

import matplotlib.pyplot as plt
import rasterio
import rasterio.warp
import shapely
from rasterio import mask
from shapely.plotting import plot_polygon

from tcd_pipeline.scripts._instance import Instance, instances_from_geo


def plot_random(crops):
    random.shuffle(crops)
    random_crops = crops[:25]

    plt.subplots(5, 5, figsize=(10, 10))
    for idx, crop in enumerate(random_crops):
        plt.subplot(5, 5, idx + 1)
        plt.imshow(crop.raster)
        plot_polygon(crop.polygon, add_points=False, edgecolor="red", facecolor="none")
        plt.axis("off")


def extract_crops(image: str, instances: list[Instance]) -> list[Instance]:
    """
    Extract crops from a source image, given a list of instances.

    **The returned instances have polygons in pixel coordinates.**
    """

    with rasterio.open(image) as src:
        # Transform geo crs -> image crs
        all_shapes = [
            rasterio.warp.transform_geom(i.crs, src.crs, i.polygon) for i in instances
        ]

        out = []
        for idx, shape in enumerate(all_shapes):
            # Limit to one geom
            shapes = [shape]

            if not (shapely.geometry.shape(shape)).within(
                shapely.geometry.box(*src.bounds)
            ):
                continue

            # Window for shape
            window = mask.geometry_window(src, shapes)
            out_crop = src.read(window=window)

            window_transform = rasterio.windows.transform(window, src.transform)
            out_shape = (window.height, window.width)
            out_mask = rasterio.mask.geometry_mask(
                shapes, out_shape=out_shape, transform=window_transform
            )

            poly_px = shapely.affinity.affine_transform(
                shapely.geometry.shape(shape), (~src.transform).to_shapely()
            )
            poly_px = shapely.affinity.translate(
                poly_px, xoff=-poly_px.bounds[0], yoff=-poly_px.bounds[1]
            )

            new_instance = Instance(
                polygon=poly_px,
                raster=out_crop[:3].transpose((1, 2, 0)),
                score=instances[idx].score,
                class_idx=instances[idx].class_idx,
            )
            out.append(new_instance)

    return out


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("geometry", help="Path to geometries")
    parser.add_argument("image", help="Path to source image")
    parser.add_argument("--output", help="Path to output folder")
    parser.add_argument("--plot", help="Plot random samples")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    instances = instances_from_geo(args.geometry)
    crops = extract_crops(args.image, instances)

    if args.plot:
        plot_random(crops)

    if args.output:
        for i, crop in enumerate(crops):
            from PIL import Image

            img = Image.fromarray(crop.raster)
            img.save(os.path.join(args.output, f"{crop.class_idx}_{i}.jpg"))


if __name__ == "__main__":
    main()
