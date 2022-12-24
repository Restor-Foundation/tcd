from tcd_pipeline.modelrunner import ModelRunner

runner = ModelRunner("config/base_semantic_segmentation.yaml")
runner.model.train()
