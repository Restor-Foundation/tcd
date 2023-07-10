import os
import shutil

import numpy as np
import rasterio

from tcd_pipeline.util import convert_to_projected

test_image_path = "data/5c15321f63d9810007f8b06f_10_00000.tif"
assert os.path.exists(test_image_path)


def test_reproject(tmpdir):

    shutil.copy(test_image_path, tmpdir)
    temp_image = os.path.join(tmpdir, os.path.basename(test_image_path))

    convert_to_projected(temp_image, inplace=False, resample=True, target_gsd_m=0.2)

    file_name, ext = os.path.splitext(os.path.basename(test_image_path))

    new_file = os.path.join(tmpdir, file_name + "_proj_20" + ext)
    assert os.path.exists(new_file)

    with rasterio.open(new_file) as fp:
        assert np.allclose(fp.res, 0.2)


def test_reproject_inplace(tmpdir):

    shutil.copy(test_image_path, tmpdir)
    temp_image = os.path.join(tmpdir, os.path.basename(test_image_path))

    convert_to_projected(temp_image, inplace=True, resample=True, target_gsd_m=0.2)

    with rasterio.open(temp_image) as fp:
        assert np.allclose(fp.res, 0.2)
