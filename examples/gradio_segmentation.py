import os

import cv2
import gradio as gr
import numpy as np
import rasterio
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

import wandb
from tcd_pipeline.modelrunner import ModelRunner
from tcd_pipeline.models.semantic_segmentation import ImageDataset

NUM_IMGS = 10

# load model
# run = wandb.init()
# artifact = run.use_artifact(
#    "dsl-ethz-restor/vanilla-model-stable/model-ukpbckms:v0", type="model"
# )
# artifact_dir = artifact.download()

runner = ModelRunner("config/base_semantic_segmentation.yaml")

# transform = T.ToPILImage()

# segmentation function for gradio interface
def segment(image, mask=None):

    # image = rasterio.open(image)

    results = runner.predict(image, warm_start=False)

    if mask:
        mask = np.load(mask)["arr_0"].astype(int)
    else:
        mask = np.full(shape=(image.shape[2], image.shape[3]), fill_value=0)

    mask = torch.from_numpy(mask)

    # Hacky probability map
    prob = cv2.cvtColor(
        cv2.applyColorMap(
            (255 * results.confidence_map).astype(np.uint8),
            colormap=cv2.COLORMAP_INFERNO,
        ),
        cv2.COLOR_RGB2BGR,
    )

    pred = cv2.cvtColor(
        cv2.applyColorMap(
            (255 * results.prediction_mask).astype(np.uint8),
            colormap=cv2.COLORMAP_INFERNO,
        ),
        cv2.COLOR_RGB2BGR,
    )

    images = {
        "true": cv2.imread(image),
        "prediction": pred,
        "probability": prob,
    }

    return [images["true"], images["prediction"], images["probability"]]


if __name__ == "__main__":

    dataset = ImageDataset(
        data_dir="data/restor-tcd-oam", setname="test", transform=None
    )

    img_list = []
    mask_list = []

    if not os.path.exists("data-demo/"):
        os.makedirs("data-demo/")

    for i in tqdm(range(NUM_IMGS + 1)):
        item = dataset.__getitem__(i)
        if item is not None:
            img_list.append(item["image"])
            save_image(item["image"] / 255.0, "data-demo/img" + str(i) + ".jpg")
            mask_list.append(item["mask"])
            np.savez_compressed("data-demo/mask" + str(i), item["mask"])
        else:
            i -= 1

    img_names = []
    mask_names = []

    for i in tqdm(range(NUM_IMGS + 1)):
        img_name = os.getcwd() + "/data-demo/img" + str(i) + ".jpg"
        mask_name = os.getcwd() + "/data-demo/mask" + str(i) + ".npz"
        if os.path.exists(img_name) and os.path.exists(mask_name):
            img_names.append(img_name)
            mask_names.append(mask_name)

    examples = [[i, m] for i, m in zip(img_names, mask_names)]

    image = gr.Image(
        shape=list(img_list[0].shape[1:3]), type="filepath", label="Original Image"
    )
    mask = gr.File(label="Mask")

    true = gr.Image(
        shape=list(img_list[0].shape[1:3]), type="pil", label="Ground Truth"
    )
    masked = gr.Image(
        shape=list(img_list[0].shape[1:3]), type="pil", label="Predicted Mask"
    )
    probs = gr.Image(
        shape=list(img_list[0].shape[1:3]), type="pil", label="Probabilities"
    )

    # create and lauch gradio interface
    demo = gr.Interface(
        fn=segment,
        inputs=[image, mask],
        outputs=[true, masked, probs],
        examples=examples,
        allow_flaggin="never",
    )

    demo.launch(share=False)

    run.finish()
