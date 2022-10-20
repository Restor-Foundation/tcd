import torchgeo
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

import os
import json
import numpy as np

import rasterio
from torch.utils.data import DataLoader
from PIL import Image
from pathlib import Path

import torch
from torchgeo.trainers import SemanticSegmentationTask

from pytorch_lightning import Trainer



# collect data and create dataset
class ImageDataset(Dataset):
    def __init__(self, setname, transform=None, target_transform=None):
        
        # Define the  mask file and the json file for retrieving images
        #self.data_dir = os.getcwd()
        self.data_dir = '../data/restor-tcd-oam/'
        self.setname = setname
        assert setname in ['train','test','val']
        #self.mask_file = np.load(self.data_dir + setname + '_masks.npz')
        #self.mask_file = np.load('data/restor-tcd-oam/' + setname + '_masks.npz')

        json_file = open(self.data_dir + setname + '_20221010.json')
        self.metadata = json.load(json_file)
        json_file.close()

        # Extract: I think we should do it directly in __getitem__
        #self.img_labels = pd.read_csv(self.mask_file)
        #self.img_dir = img_dir (input?)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(list(self.metadata.items())[2][1])

    def __getitem__(self, idx):
        img_name = list(self.metadata.items())[2][1][idx]['file_name']
        img_path = os.path.join(self.data_dir, 'images', img_name)
        #img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        #image = np.array(Image.open(img_path))
        image = torch.Tensor(np.array(Image.open(img_path)))
        image = torch.permute(image, (2,0,1))
    
        #mask_path = os.path.join(self.data_dir, self.setname + '_masks.npz')
        mask = np.load(self.data_dir + 'masks/' + self.setname + '_mask_' + idx + '.npz')#['arr_0']
        #mask = self.mask_file['arr_0'][idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return {'image':image, 'mask':mask}

#
setname = 'train'
train_data = ImageDataset(setname)
setname = 'val'
val_data = ImageDataset(setname)
setname = 'test'
test_data = ImageDataset(setname)

#DataLoader
train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=True)

#set up task
task = SemanticSegmentationTask(
    segmentation_model='unet',
    encoder_name='resnet18',
    encoder_weights='imagenet',
    in_channels=3, # no infra red
    num_classes=2,
    loss='ce',
    ignore_index=None,
    learning_rate=0.1,
    learning_rate_schedule_patience=5
)

trainer = Trainer() # add kwargs later
trainer.fit(task, train_dataloader, val_dataloader)

