import logging
import os

from tcd_pipeline.post_processing import ProcessedResult

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

image_path = "./data/restor-tcd-oam/images/5b1096a62b6a08001185f4cf_10_00030.tif"
results_file = "./data/restor-tcd-oam/train_20221010.json"

output_path = os.path.join(
    "data",
    os.path.splitext(os.path.basename(image_path))[0] + "_pred",
)

os.makedirs(output_path, exist_ok=True)

results = ProcessedResult.load_serialisation(results_file, image_path, global_mask=True)

logger.info("Saving prediction masks")
for threshold in [0.2, 0.4, 0.6, 0.8]:
    results.set_threshold(threshold)
    results.save_masks(output_path, image_path=image_path, suffix=f"_{str(threshold)}")

logger.info("Saving shapefile")
# results.set_threshold(0.4)
# results.save_shapefile(os.path.join(output_path, "instances_40.shp"), image_path)

logger.info("Saving visualised predictions")
results.visualise(output_path=os.path.join(output_path, "predictions.png"))
