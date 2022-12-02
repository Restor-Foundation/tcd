import logging

from model import TiledModel

logger = logging.getLogger("__name__")


class SemanticSegmentationModel(TiledModel):
    def __init__(self, config):
        super().__init__(config)

    def load_model(self):
        pass

    def train(self):
        pass

    def evaluate(self, dataset, output_folder):
        pass

    def predict(self, image):
        pass

    def on_after_predict(self, results):
        pass

    def post_process(self):
        pass
