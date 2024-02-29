import hydra
from omegaconf import DictConfig

from tcd_pipeline.modelrunner import ModelRunner


@hydra.main(
    version_base=None, config_path="src/tcd_pipeline/config", config_name="config"
)
def main(cfg: DictConfig):
    runner = ModelRunner(cfg)

    if cfg.job == "train":
        runner.train()
    elif cfg.job == "predict":
        runner.predict(cfg.input)


if __name__ == "__main__":
    main()
