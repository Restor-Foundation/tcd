import argparse
import os

from tcd_pipeline.modelrunner import ModelRunner

os.environ["WANDB_MODE"] = "disabled"

parser = argparse.ArgumentParser()
parser.add_argument("config", nargs="?", default="config/train_detectron.yaml")
args = parser.parse_args()

runner = ModelRunner(args.config)
runner.train()
