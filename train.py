import hydra
from omegaconf import DictConfig

from tcd_pipeline import Pipeline


@hydra.main(
    version_base=None, config_path="src/tcd_pipeline/config", config_name="config"
)
def main(cfg: DictConfig):
    runner = Pipeline(cfg)
    res = runner.train()


if __name__ == "__main__":
    main()
