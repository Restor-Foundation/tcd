import hydra
from omegaconf import DictConfig

from tcd_pipeline.modelrunner import ModelRunner


@hydra.main(
    version_base=None, config_path="src/tcd_pipeline/config", config_name="config"
)
def main(cfg: DictConfig):
    runner = ModelRunner(cfg)
    runner.train()


if __name__ == "__main__":
    main()
