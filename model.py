import gc
import logging
import os
import time
from datetime import datetime

import dotmap
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import wandb
import yaml
from detectron2 import model_zoo
from detectron2.config import CfgNode, get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor, DefaultTrainer, HookBase
from detectron2.evaluation import COCOEvaluator
from detectron2.modeling.test_time_augmentation import GeneralizedRCNNWithTTA
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode, Visualizer
from PIL import Image
from pydantic import NoneStr
from torch.utils.data import DataLoader, Dataset
from torchgeo.datasets import RasterDataset, stack_samples
from torchgeo.samplers import GridGeoSampler
from tqdm.auto import tqdm

logger = logging.getLogger("__name__")


class Trainer(DefaultTrainer):
    """
    Subclass of the default training class
    so that we have control over things like
    augmentation.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, tasks=["segm"], output_dir=cfg.OUTPUT_DIR)


class TrainExampleHook(HookBase):
    def log_image(self, image, key, caption=""):
        images = wandb.Image(image, caption)
        wandb.log({key: images})

    def before_train(self):

        data = self.trainer.data_loader
        batch = next(iter(data))[:5]
        resize = torchvision.transforms.Resize(512)
        bgr_permute = [2, 1, 0]

        # Cast to float here, otherwise torchvision complains
        image_grid = torchvision.utils.make_grid(
            [resize(s["image"].float()[bgr_permute, :, :]) for s in batch],
            value_range=(0, 255),
            normalize=True,
        )
        self.log_image(
            image_grid, key="train_examples", caption="Sample training images"
        )

    def after_step(self):
        pass


class ModelRunner:
    """Class for running instance segmentation tasks (train/eval/predict)"""

    def __init__(self, config):
        """Initialise model runner

        Args:
            config (str, dict): Config file or path to config file
        """

        if isinstance(config, str):
            with open(config, "r") as fp:
                config = yaml.safe_load(fp)

        self.setup(config)
        self.model = None
        self.should_reload = False

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Running inference using: {self.device}")

    def train(self):
        """Initiate model training, uses configuration"""

        os.makedirs(self.config.data.output, exist_ok=True)

        # Ensure that this gets run before any serious
        # PyTorch stuff happens (but after config is fine)
        if wandb.run is None:
            wandb.tensorboard.patch(root_logdir=self.config.data.output, pytorch=True)
            wandb.init(
                config=self.config,
                settings=wandb.Settings(start_method="thread", console="off"),
            )

        # Detectron starts tensorboard
        setup_logger()

        from detectron2.data.datasets import register_coco_instances

        register_coco_instances(
            "train", {}, self.config.data.train, self.config.data.images
        )
        register_coco_instances(
            "test", {}, self.config.data.test, self.config.data.images
        )
        register_coco_instances(
            "validate", {}, self.config.data.validation, self.config.data.images
        )

        import torch.multiprocessing

        torch.multiprocessing.set_sharing_strategy("file_system")

        gc.collect()

        if "cuda" in self.device:
            with torch.no_grad():
                torch.cuda.empty_cache()

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(self.config.model.architecture))
        cfg.merge_from_other_cfg(CfgNode(self.config.evaluate.detectron))

        # Checkpoint
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )

        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.config.data.classes)

        cfg.MODEL.DEVICE = self.device

        cfg.DATASETS.TRAIN = "train"
        cfg.DATASETS.TEST = "validate"

        cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
        cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 1000
        cfg.TEST.EVAL_PERIOD = cfg.SOLVER.MAX_ITER // 10
        cfg.SOLVER.CHECKPOINT_PERIOD = cfg.SOLVER.MAX_ITER // 5
        cfg.SOLVER.BASE_LR = 1e-3
        cfg.SOLVER.STEPS = (
            int(cfg.SOLVER.MAX_ITER * 0.75),
            int(cfg.SOLVER.MAX_ITER * 0.9),
        )

        now = datetime.now()  # current date and time
        cfg.OUTPUT_DIR = os.path.join(self.config.data.output, now.strftime("%Y%m%d"))
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

        trainer = Trainer(cfg)

        example_logger = TrainExampleHook()
        trainer.register_hooks([example_logger])
        trainer.resume_or_load(resume=False)

        """
        Run summary information for debugging/tracking, contains:

        - dataset sizes 

        """
        wandb.run.summary["train_size"] = len(DatasetCatalog.get("train"))
        wandb.run.summary["test_size"] = len(DatasetCatalog.get("test"))
        wandb.run.summary["val_size"] = len(DatasetCatalog.get("validate"))

        trainer.train()

    def preprocess(self):
        pass

    def postprocess(self):
        pass

    def load_model(self):

        import torch.multiprocessing

        torch.multiprocessing.set_sharing_strategy("file_system")

        gc.collect()

        if "cuda" in self.device:
            with torch.no_grad():
                torch.cuda.empty_cache()

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(self.config.model.architecture))
        cfg.merge_from_other_cfg(CfgNode(self.config.evaluate.detectron))
        cfg.MODEL.WEIGHTS = self.config.model.weights
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.config.data.classes)

        cfg.MODEL.DEVICE = self.device

        _cfg = cfg.clone()

        self.predictor = DefaultPredictor(_cfg)

        if self.config.evaluate.detectron.TEST.AUG.ENABLED:
            self.model = GeneralizedRCNNWithTTA(
                _cfg, self.predictor.model, batch_size=6
            )
        else:
            self.model = self.predictor.model

        MetadataCatalog.get(
            self.config.data.name
        ).thing_classes = self.config.data.classes
        self.num_classes = len(self.config.data.classes)
        self.max_detections = self.config.evaluate.detectron.TEST.DETECTIONS_PER_IMAGE

    def predict_file(self, filename):
        """Run inference on a single file

        Args:
            filename (str): Path to image

        Returns:
            predictions: Dictionary with prediction results
        """

        image = np.array(Image.open(filename))
        image_tensor = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        return self.predict_tensor(image_tensor)

    def attempt_reload(self, timeout_s=60):

        if "cuda" not in self.device:
            return

        self.should_exit = False
        tstart = time.time()
        delay_s = 2

        while True:

            telapsed = time.time() - tstart

            if telapsed > timeout_s:
                logger.error("Timeout.")
                # self.should_exit = True
                return

            torch.cuda.synchronize()

            try:
                del self.model
                del self.predictor
            except:
                pass

            self.model = None
            self.predictor = None

            with torch.no_grad():
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

            try:
                logger.error(f"Reloading model ({telapsed:1.2f} s)")
                self.load_model()
            except Exception as e:
                logger.error(f"Failed to restart model. {e}")

            # Ignore first traceback?
            try:
                1 / 0
            except:
                pass

            try:
                logger.warning(f"Attempting to run test inference")
                x = torch.rand((3, 2048, 2048)).float()
                preds = self.predict_tensor(x)

                if preds != None:
                    self.should_reload = False
                    break

            except Exception as e:
                logger.error(f"Failed to run model, delaying {e}")

            time.sleep(delay_s)
            delay_s *= 2

    def predict_tensor(self, image_tensor):
        """Run inference on an image tensor

        Args:
            image_tensor (torch.Tensor): Float tensor in CHW order, un-normalised

        Returns:
            predictions: Detectron2 prediction dictionary
        """

        if self.model is None:
            self.load_model()

        self.model.eval()
        self.should_reload = False
        predictions = None

        with torch.no_grad():
            _, height, width = image_tensor.shape

            inputs = {"image": image_tensor, "height": height, "width": width}

            try:
                predictions = self.model([inputs])[0]["instances"]

                if len(predictions) >= self.max_detections:
                    logger.warning(
                        f"Maximum detections reached ({self.max_detections}), possibly re-run with a higher threshold."
                    )

            except RuntimeError as e:
                logger.error(f"Runtime error: {e}")
                self.should_reload = True
            except Exception as e:
                logger.error(
                    f"Failed to run inference: {e}. Attempting to reload model."
                )
                self.should_reload = True

        return predictions

    @classmethod
    def dataloader_from_image(self, image_path, tile_size, stride):
        """Generate a torchgeo dataloader from a single (potentially large) image.

        This function is a convenience utility that creates a dataloader for tiled
        inference. Essentially it subclasses RasterDataset with a glob that locates
        a single image (based on the given image path).

        Args:
            image_path (str): Path to image
            tile_size (int): Tile size, chosen to accommodate available VRAM.
            stride (int): Stride, nominally around 1 receptive field

        Returns:
            _type_: _description_
        """
        filename = os.path.basename(image_path)
        dirname = os.path.dirname(image_path)

        class SingleImageRaster(RasterDataset):
            filename_glob = filename
            is_image = True
            separate_files = False

        dataset = SingleImageRaster(root=dirname)
        sampler = GridGeoSampler(dataset, size=tile_size, stride=stride)
        dataloader = DataLoader(
            dataset, batch_size=1, sampler=sampler, collate_fn=stack_samples
        )

        return dataloader

    def detect_tiled(
        self,
        image_path,
        tile_size=1024,
        pad=512,
        confidence_thresh=0.5,
        skip_empty=True,
    ):
        """Run inference on an image using tiling

        The output from this function is a list of predictions per-tile. Each output is a standard detectron2 result
        dictionary with the associated geo bounding box. This can be used to geo-locate the predictions, or to map
        to the original image.

        Args:
            image_path (str): Path to image file
            tile_size (int, optional): Tile size. Defaults to 1024.
            pad (int, optional): Amount to pad around image when tiling (e.g. stride). Defaults to 512.
            confidence_thresh (float, optional): Confidence threshold for predictions. Defaults to 0.5.
            skip_empty (bool, optional): Skip empty/all-black images. Defaults to True.

        Returns:
            list(tuple(prediction, bounding_box)): A list of detectron2 predictions and the bounding boxes for those detections.
        """

        dataloader = self.dataloader_from_image(
            image_path, tile_size, stride=tile_size - pad
        )

        pbar = tqdm(dataloader, total=len(dataloader))
        results = []
        self.failed_images = []
        self.should_exit = False

        # Predict on each tile
        for batch in pbar:

            if self.should_exit:
                break

            if self.should_reload:
                self.attempt_reload()

            if "cuda" in self.device:
                free_memory_b, used_memory_b = torch.cuda.mem_get_info()

            image = batch["image"][0].float()

            if image.mean() < 1 and skip_empty:
                if "cuda" in self.device:
                    pbar.set_postfix_str(
                        f"Memory: {free_memory_b/1073741824:1.2f}/{used_memory_b/1073741824:1.2f}, Empty frame"
                    )
                continue

            predictions = self.predict_tensor(image)

            # Typically if this happens we hit an OOM...
            if predictions is None:
                pbar.set_postfix_str("Error")
                logger.error("Failed to run inference on image.")
                self.failed_images.append(image)
            else:
                if "cuda" in self.device:
                    pbar.set_postfix_str(
                        f"Memory used: {used_memory_b/1073741824:1.2f}, Instances: {len(predictions)}"
                    )
                results.append((predictions, batch["bbox"][0]))

            del predictions

        return results

    @classmethod
    def bbox_to_original_image(self, bbox, image):
        """Convert bounding box coords to image coords (px)

        Args:
            bbox (_type_): Detectron2 bounding box
            image (rasterio image): Rasterio image

        Returns:
            _type_: _description_
        """

        miny, minx = image.index(bbox.minx, bbox.miny)
        maxy, maxx = image.index(bbox.maxx, bbox.maxy)

        return minx, miny, maxx, maxy

    def check_offsets(results, image):
        for result in results:
            print(
                f"Offset (m) {result[1].minx-image.bounds.left:1.2f} {result[1].miny-image.bounds.bottom:1.2f}"
            )
            print(image.index(result[1].minx, result[1].miny)[::-1])

    def merge_tiled_results(self, results, image, threshold=0.5):
        """Merge tiled results into a single mask

        Return results as a flat mask. This is convenient because it effectively
        merges overlapping instances (which can be separated back out by contour
        detection if necessary). Good for quick visuallisation.

        Output is an segmentation array the same size as the original image where each channel
        corresponds to a class.

        Args:
            results (list(tuple(prediction, boundingbox))): List of predictions and torchgeo bounding boxes
            image (rasterio image): Input image (that was tiled)
            threshold: Confidence threshold

        Returns:
            image mask: HxWxM array
        """

        image_mask = np.zeros(
            (image.height, image.width, self.num_classes), dtype=np.uint8
        )

        for i, result in enumerate(results):

            instances, bbox = result
            minx, miny, maxx, maxy = self.bbox_to_original_image(bbox, image)

            # Sort coordinates if necessary
            if miny > maxy:
                miny, maxy = maxy, miny

            if minx > maxx:
                minx, maxx = maxx, minx

            for i in range(len(instances)):

                if instances.scores[i] < threshold:
                    continue

                mask = instances.pred_masks[i].cpu().numpy()
                class_idx = instances.pred_classes[i]
                mask_height, mask_width = mask.shape

                image_mask[
                    miny : miny + mask_height, minx : minx + mask_width, class_idx
                ][mask] = 1

        return image_mask

    def visualise(self, image, results, confidence_thresh=0.65, **kwargs):

        mask = results.scores > confidence_thresh
        v = Visualizer(
            image,
            MetadataCatalog.get(self.config.data.name),
            scale=1.2,
            instance_mode=ColorMode.SEGMENTATION,
        )
        out = v.draw_instance_predictions(results[mask].to("cpu"))

        plt.figure(**kwargs)
        plt.imshow(out.get_image())

    def evaluate(self, dataset, output_folder):

        if self.model is not None:
            self.load_model()

        test_loader = build_detection_test_loader(_cfg, dataset_test, batch_size=1)

        os.makedirs(eval_output_dir, exist_ok=True)
        evaluator = COCOEvaluator(
            dataset_name=dataset,
            tasks=["segm"],
            distributed=False,
            output_dir=output_folder,
            max_dets_per_image=500,
            allow_cached_coco=False,
        )

        # Run the evaluation
        inference_on_dataset(model, test_loader, evaluator)

    def setup(self, config):
        self.config = dotmap.DotMap(config)
