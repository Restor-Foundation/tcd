import numpy as np
import rasterio
from shapely.geometry import Polygon, box


def random_polygon(bounding_box: box) -> Polygon:
    """
    Generate a random simple polygon within a given bounding box.

    Args:
    - bounding_box (shapely.geometry.box): The bounding box within which the polygon will be generated.

    Returns:
    - shapely.geometry.Polygon: A simple polygon that fits within the given bounding box.
    """

    # Randomly decide how many vertices the polygon will have (at least 3, up to a maximum of 10 for simplicity)
    num_vertices = np.random.randint(3, 11)

    # Generate random points within the bounding box
    minx, miny, maxx, maxy = bounding_box.bounds
    points = np.random.rand(num_vertices, 2)
    points[:, 0] = (points[:, 0] * (maxx - minx)).astype(int) + minx
    points[:, 1] = (points[:, 1] * (maxy - miny)).astype(int) + miny

    # Use a convex hull to ensure the polygon is simple (no self-intersecting edges)
    polygon_points = Polygon(points).convex_hull

    return polygon_points


def random_bbox(
    image: rasterio.DatasetReader, min_width: int, min_height: int
) -> Polygon:
    """
    Generate a random bounding box within the bounds of a raster image dataset.

    Args:
    - image (rasterio.DatasetReader): The raster image dataset.
    - min_width (int): The minimum width of the bounding box.
    - min_height (int): The minimum height of the bounding box.

    Returns:
    - shapely.geometry.box: A bounding box that fits within the image's dimensions.
    """
    # Image dimensions
    img_width, img_height = image.width, image.height

    # Ensure the minimum dimensions do not exceed the image dimensions
    min_width = min(min_width, img_width)
    min_height = min(min_height, img_height)

    # Calculate maximum starting points for the random box
    max_x_start = img_width - min_width
    max_y_start = img_height - min_height

    # Generate random starting points
    x_start = np.random.randint(0, max_x_start + 1)
    y_start = np.random.randint(0, max_y_start + 1)

    # Randomly decide the actual width and height of the box, ensuring it stays within the image
    width = np.random.randint(min_width, img_width - x_start + 1)
    height = np.random.randint(min_height, img_height - y_start + 1)

    # Create the bounding box
    return box(x_start, y_start, x_start + width, y_start + height)
