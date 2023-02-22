import argparse
import os
import tempfile

import rasterio
from fastapi import FastAPI, File, Form, UploadFile

from tcd_pipeline.modelrunner import ModelRunner

app = FastAPI()
app.runner = ModelRunner("../config/segmentation_tta.yaml")
app.mode = "semantic"
configs = {
    "semantic": "../config/base_semantic_segmentation.yaml",
    "semantic_tta": "../config/segmentation_tta.yaml",
    "instance": "../config/base_detectron.yaml",
    "instance_tta": "../config/detectron_tta.yaml",
}


def switch_mode(app, mode, tta):

    if mode not in configs.keys():
        raise NotImplementedError("Mode should be in [instance, semantic].")

    if tta:
        mode += "_tta"

    if app.mode != mode:
        app.runner = ModelRunner(configs[mode])
        app.mode = mode


@app.post("/predict")
def predict(
    file: UploadFile = File(...),
    mode: str = "semantic",
    tta: bool = True,
    report: bool = False,
    instances: bool = True,
):

    results = {}

    try:
        contents = file.file.read()
        with rasterio.open(file.filename) as src:

            switch_mode(app, mode, tta)

            res = app.runner.predict(src, warm_start=False)

            if mode == "instance":
                results["tree_mask"] = res.tree_mask.tobytes()

                if instances:
                    results["instances"] = res.serialise()

            canopy_mask = res.canopy_mask
            results["canopy_mask"] = canopy_mask.tobytes()
            results["shape"] = canopy_mask.shape
            results["dtype"] = "bool"

            if report:
                raise NotImplementedError("Report generation is not implemented yet.")

    except Exception as e:
        return {"success": False, "error": e}
    finally:
        file.file.close()

    return {"success": True, "results": results}
