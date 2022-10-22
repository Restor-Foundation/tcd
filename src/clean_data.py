import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

from PIL import UnidentifiedImageError


DATA_DIR = "data/images/"
C_DATA_DIR = "data/corrupted_images/"


#try to open all images
ctr = 0
print('checking for corrupted images')
for filename in tqdm(os.listdir(DATA_DIR)):
    try:
        im=Image.open(DATA_DIR+filename)
    except UnidentifiedImageError:
        ctr += 1
        if not os.path.exists(C_DATA_DIR):
            print('creating folder for corrupted images')
            os.makedirs(C_DATA_DIR)

        os.rename(DATA_DIR+filename, C_DATA_DIR+filename)
print('Found and moved ' + str(ctr) + ' corrupted images')