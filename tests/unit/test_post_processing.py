import os
import tempfile

import numpy as np
import pytest
import rasterio

from tcd_pipeline.general_statistics import Statistics
from tcd_pipeline.post_processing import InstanceSegmentationResult

image_path = "data/5c15321f63d9810007f8b06f_10_00000.tif"
results_file = "tests/unit/5c15321f63d9810007f8b06f_10_00000.json"
threshold = 0.8


@pytest.fixture(scope="session")
def serialised_result():
    return InstanceSegmentationResult.load_serialisation(
        results_file, image_path, global_mask=False
    )


def test_load_serialisation(serialised_result):
    assert len(serialised_result.get_trees()) > 0


def test_make_masks_georef(serialised_result, tmpdir):
    serialised_result.save_masks(
        output_path=tmpdir, image_path=image_path, suffix=f"_{str(threshold)}"
    )

    assert os.path.exists(os.path.join(tmpdir, "tree_mask_0.8.tif"))
    assert os.path.exists(os.path.join(tmpdir, "canopy_mask_0.8.tif"))


def test_make_masks(serialised_result, tmpdir):
    serialised_result.save_masks(output_path=tmpdir, suffix=f"_{str(threshold)}")

    assert os.path.exists(os.path.join(tmpdir, "tree_mask_0.8.tif"))
    assert os.path.exists(os.path.join(tmpdir, "canopy_mask_0.8.tif"))


def test_visualise(serialised_result, tmpdir):
    output_path = os.path.join(tmpdir, "predictions.png")
    serialised_result.visualise(output_path=output_path)
    assert os.path.exists(output_path)


def test_shapefile(serialised_result, tmpdir):
    serialised_result.save_shapefile(
        output_path=os.path.join(tmpdir, "test.shp"), image_path=image_path
    )

    assert os.path.exists(os.path.join(tmpdir, "test.shp"))
    assert os.path.exists(os.path.join(tmpdir, "test.prj"))
    assert os.path.exists(os.path.join(tmpdir, "test.cpg"))
    assert os.path.exists(os.path.join(tmpdir, "test.dbf"))
    assert os.path.exists(os.path.join(tmpdir, "test.shx"))


def test_geometry_filter(serialised_result):

    # Old numbers for comparison
    num_pixels_prev = serialised_result.num_valid_pixels
    bounds = serialised_result.image.bounds
    prev_trees = serialised_result.get_trees()

    # New bounds that are in the centre of the image
    width = bounds.right - bounds.left
    height = bounds.top - bounds.bottom
    new_bounds = rasterio.coords.BoundingBox(
        left=bounds.left + 0.25 * width,
        right=bounds.right - 0.25 * width,
        top=bounds.top - 0.25 * height,
        bottom=bounds.bottom + 0.25 * height,
    )

    # Geometry corresponding to updated bounds
    from shapely.geometry import box

    geom = box(*new_bounds)

    # Apply bounds
    serialised_result.set_roi(geom)

    assert np.allclose(0.25, serialised_result.num_valid_pixels / num_pixels_prev)
    assert serialised_result.tree_cover > 0
    assert serialised_result.canopy_cover > 0
    assert len(serialised_result.get_trees()) < len(prev_trees)


def test_statistics(serialised_result):
    stats = Statistics()
    res = stats.run(serialised_result)
