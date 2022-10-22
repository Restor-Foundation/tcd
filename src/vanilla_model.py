import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"  # quick fix, see below

import json
from pathlib import Path

import argparse
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
import torchgeo
from PIL import Image
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, Dataset
from torchgeo.trainers import SemanticSegmentationTask
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--setup', type=bool, required=True)
args = parser.parse_args()

if args.setup == True:
    import clean_data

# if the imports throw OMP error #15, try $ conda install nomkl
# or, as an unsafe quick fix like above, import os; os.environ['KMP_DUPLICATE_LIB_OK']='True';


# collect data and create dataset
class ImageDataset(Dataset):
    def __init__(self, setname, transform=None, target_transform=None):

        # Define the  mask file and the json file for retrieving images
        # self.data_dir = os.getcwd()
        self.data_dir = '/cluster/scratch/earens/data/' #data/
        self.setname = setname
        assert setname in ["train", "test", "val"]

        json_file = open(self.data_dir + setname + "_20221010.json")
        self.metadata = json.load(json_file)
        json_file.close()

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(list(self.metadata.items())[2][1])

    def __getitem__(self, idx):
        try:
            img_name = list(self.metadata.items())[2][1][idx]["file_name"]
            img_path = os.path.join(self.data_dir, "images", img_name)
            image = torch.Tensor(np.array(Image.open(img_path)))
            image = torch.permute(image, (2, 0, 1))

            mask = np.load(
                self.data_dir + "masks/" + self.setname + "_mask_" + str(idx) + ".npz"
            )['arr_0'].astype(int)

            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                mask = self.target_transform(mask)
            return {"image": image, "mask": mask}
        except: self.__getitem__(idx)


# be aware that the following crashes on CPU with 16GB RAM, work in progress

if __name__ == "__main__":

    # create datasets
    setname = "train"
    train_data = ImageDataset(setname)
    setname = "val"
    val_data = ImageDataset(setname)
    setname = "test"
    test_data = ImageDataset(setname)

    # DataLoader
    train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=2)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=2)

    # set up task
    task = SemanticSegmentationTask(
        segmentation_model="unet",
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=3,  # no infra red
        num_classes=2,
        loss="ce",
        ignore_index=None,
        learning_rate=0.1,
        learning_rate_schedule_patience=5,
    )

    trainer = Trainer(accelerator="gpu", max_epochs = 50)  # add kwargs later
    trainer.fit(task, train_dataloader, val_dataloader)   

