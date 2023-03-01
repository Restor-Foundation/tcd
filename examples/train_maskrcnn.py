import os

from tcd_pipeline.modelrunner import ModelRunner

os.environ["WANDB_MODE"] = "disabled"

runner = ModelRunner("config/train_detectron.yaml")
runner.train()
