import os
import tempfile

import pytest

from tcd_pipeline.general_statistics import Statistics
from tcd_pipeline.post_processing import ProcessedResult

# TODO this adds a lot of time onto the test suite, probably should use a
# simpler dedicated test example.
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


def test_statistics(serialised_result):
    stats = Statistics()
    res = stats.run(serialised_result)
