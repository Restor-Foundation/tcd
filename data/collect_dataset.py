import glob
import json
import os
import subprocess

import dotenv
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

dotenv.load_dotenv()

downloaded_images = set(glob.glob("./images/*.tif"))

filenames = lambda data: set(
    [os.path.join("./images/", image["file_name"]) for image in data["images"]]
)

with open("test_20221010.json") as fp:
    data = json.load(fp)
test_images = filenames(data)

with open("train_20221010.json") as fp:
    data = json.load(fp)
train_images = filenames(data)

with open("val_20221010.json") as fp:
    data = json.load(fp)
val_images = filenames(data)

all_images = test_images.union(train_images).union(val_images)

for image in tqdm(all_images):
    try:
        _ = np.array(Image.open(image))
        continue
    except Exception as e:
        image_id = image.split("/")[4].split("_")[0]
        base = os.path.basename(image)
        subprocess.call(
            [
                "gsutil",
                "-m",
                "cp",
                f"gs://{os.get_env('GCP_BUCKET')}/image/oam/id/{image_id}/tiles/{base}",
                "./images",
            ]
        )

    try:
        _ = np.array(Image.open(image))
    except:
        print(f"Still failed to open {image}")
