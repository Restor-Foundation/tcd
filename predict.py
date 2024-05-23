import os

import hydra
from omegaconf import DictConfig

from tcd_pipeline import Pipeline


@hydra.main(
    version_base=None, config_path="src/tcd_pipeline/config", config_name="config"
)
def main(cfg: DictConfig):
    runner = Pipeline(cfg)
    res = runner.predict(cfg.input)

    res.serialise(cfg.output)
    res.save_masks(cfg.output)

    if cfg.model.type == "instance_segmentation":
        res.save_shapefile(os.path.join(cfg.output, "instances.shp"))


if __name__ == "__main__":
    main()
