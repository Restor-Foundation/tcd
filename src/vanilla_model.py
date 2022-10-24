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

def str2bool(input):
    if input == 'true':
        return True
    elif input == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument("--setup", type=str2bool, nargs='?', const=True, default=False, choices=['true', 'false'],
                    help="Required for initial data preparation")
parser.add_argument("--lr", type=float, nargs='?', const=True, default=0.01,
                    help="Set learning rate for training")
parser.add_argument("--model", type=str, nargs='?', const=True, default='deeplabv3+', choices=['deeplabv3+', 'unet','fcn'],
                    help="Choose model")
parser.add_argument("--backbone", type=str, nargs='?', const=True, default='resnet18', choices=['resnet50', 'resnet18','resnet34'],
                    help="Choose backbone")
parser.add_argument("--epochs", type=int, nargs='?', const=True, default= 1000,
                    help="Set max nr of epochs")
parser.add_argument("--loss", type=str, nargs='?', const=True, default= 'ce',choices=['ce', 'jaccard','focal'],
                    help="Choose loss function")
parser.add_argument("--pretrained", type=str2bool, nargs='?', const=True, default= True,
                    help="Optional use of pretrained weights")
parser.add_argument("--scale", type=float, nargs='?', const=True, default= 1,
                    help="Choose downsampling factor")
args = parser.parse_args()


if args.setup:
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
        img_name = list(self.metadata.items())[2][1][idx]["file_name"]
        img_path = os.path.join(self.data_dir, "images", img_name)
        try:
            image = torch.Tensor(np.array(Image.open(img_path)))
        except:
            return None
        image = torch.permute(image, (2, 0, 1))

        mask = np.load(
            self.data_dir + "masks/" + self.setname + "_mask_" + str(idx) + ".npz"
        )['arr_0'].astype(int)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        return {"image": image, "mask": mask}

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

if __name__ == "__main__":

    # create datasets
    setname = "train"
    train_data = ImageDataset(setname)
    setname = "val"
    val_data = ImageDataset(setname)
    setname = "test"
    test_data = ImageDataset(setname)

    # DataLoader
    train_dataloader = DataLoader(train_data, batch_size=2, shuffle=True, num_workers=2,collate_fn=collate_fn)
    val_dataloader = DataLoader(val_data, batch_size=2, shuffle=False, num_workers=2,collate_fn=collate_fn)
    test_dataloader = DataLoader(test_data, batch_size=2, shuffle=False, num_workers=2,collate_fn=collate_fn)

    # set up task
    task = SemanticSegmentationTask(
        segmentation_model=args.model,
        encoder_name= args.backbone,
        encoder_weights= 'imagenet' if args.pretrained else 'None',
        in_channels=3,  # no infra red
        num_classes=2,
        loss= args.loss,
        ignore_index= None,
        learning_rate= args.lr,
        learning_rate_schedule_patience=5,
    )

    trainer = Trainer(accelerator="gpu", max_epochs = args.epochs, max_time="00:23:59:00") 
    trainer.fit(task, train_dataloader, val_dataloader)   

