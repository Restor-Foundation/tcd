import hydra
from omegaconf import DictConfig

from tcd_pipeline.modelrunner import ModelRunner


@hydra.main(
    version_base=None, config_path="src/tcd_pipeline/config", config_name="config"
)
def main(cfg: DictConfig):
    runner = ModelRunner(cfg)
    res = runner.predict(cfg.input)
    res.serialise(cfg.output)
    res.save_masks(cfg.output)


if __name__ == "__main__":
    main()
