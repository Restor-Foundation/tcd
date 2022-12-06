import os
import tempfile

import pytest

from tcd_pipeline.post_processing import ProcessedResult

image_path = "./data/restor-tcd-oam/images/5b1096a62b6a08001185f4cf_10_00030.tif"
results_file = "./data/restor-tcd-oam/train_20221010.json"
threshold = 0.8


@pytest.fixture(scope="session")
def serialised_result():
    return ProcessedResult.load_serialisation(
        results_file, image_path, global_mask=True
    )


def test_load_serialisation():
    results = ProcessedResult.load_serialisation(
        results_file, image_path, global_mask=True
    )
    assert len(results.get_trees()) > 0


def test_make_masks(serialised_result, tmpdir):
    serialised_result.save_masks(output_path=tmpdir, suffix=f"_{str(threshold)}")


def test_visualise(serialised_result, tmpdir):
    serialised_result.visualise(output_path=os.path.join(tmpdir, "predictions.png"))


def test_shapefile(serialised_result, tmpdir):
    serialised_result.save_shapefile(
        output_path=os.path.join(tmpdir, "test.shp"), image_path=image_path
    )
