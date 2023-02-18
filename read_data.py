import os
import sys
import copy
import random
import tqdm.notebook as tq

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.utils
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms.functional as TF

# from google.colab import drive
# drive.mount('/content/drive')

images_path = './seep_prediction/train_images_256/'
mask_path = './seep_prediction/train_masks_256/'

img_files = []
msk_files = []
for path, subdirs, files in os.walk(images_path):
  for name in files:
    img_files.append(os.path.join(images_path,name))
    msk_files.append(os.path.join(mask_path,name))


def image_properties(img):
    '''
    Print image file properties
    '''
    print('image type:', type(img))
    print('image mode:', img.mode)
    print('image info:', img.info)
    array = np.array(img)
    print('image shape:', array.shape)
    print('min:', array.min(),'std:', array.std(),'mean:', array.mean(), 'max:',  array.max())
    print('array:', array)

def preview(images):
    '''
    Preview images stacked horizontally
    '''
    fig, axs = plt.subplots(1, len(images), figsize=(10,10))
    for i, image in enumerate(images):
        axs[i].imshow(image)
        axs[i].axis('off')
    plt.show()

# We need to convert L to P(palette mode) to view the mask images nicely 
def palette():
    '''
    Returns palette (used in 'P' mode of image representation)
    '''
    colors_dict = {
        0: [0, 0, 0],       # 0 = background: black
        1: [255, 0, 0],     # 1 = seep class 1: red
        2: [255, 127, 0],   # 2: orange 
        3: [255, 255, 0],   # 3: yellow
        4: [0, 255, 0],     # 4: green
        5: [0, 0, 255],     # 5: blue
        6: [46, 43, 95],    # 6: dark blue
        7: [139, 0, 255],   # 7: purple
        }
    palette = []
    for i in np.arange(256):
        if i in colors_dict:
            palette.extend(colors_dict[i])
        else:
            palette.extend([0, 0, 0])
    return palette

def mask2p(msk):
    """
    Converts mask to P-mode image
    """
    msk=msk.convert('P')
    msk.putpalette(palette())
    return msk

def mask2onehot(mask):
    """
    Converts a segmentation mask (H,W) to (C,H,W) where the 0 dim is a C-one-hot encoding vector
    Where K - number of classes
    """
    classes = [0,1,2,3,4,5,6,7] # Classes in the dataset
    mask = np.asanyarray(mask)
        
    _mask = [mask == i for i in classes]
    mask = np.array(_mask).astype(np.uint8)
      
    #mask = np.where(mask == 0, 0, 1).astype(np.uint8) # for 1 channel only
    return mask

def onehot2mask(onehot):
    """
    Converts onehot representation (C, H, W) to mask representation (H,W)
    """
    zeros = np.zeros_like(onehot[0,:,:])
    zeros = np.expand_dims(zeros, axis=0)
    onehot = np.concatenate((zeros, onehot), axis=0)    
    array = np.argmax(onehot, axis=0).astype(np.uint8)
    
    #array = onehot.astype(np.uint8) # for 1 channel

    return Image.fromarray(array, 'L')

def onehot2p(onehot):
    """
    Converts onehot representation (C, H, W) to P image mask representation (H,W)
    """
    mask = onehot2mask(onehot)
    maskp = mask2p(mask)
    return maskp

imgs_train, imgs_val, msks_train, msks_val = \
    train_test_split(img_files, msk_files, test_size=0.2, shuffle=True, random_state=42)

# Images need to be normalized, let's find normalization parameters:
lst = []
for image in imgs_train:
    image = Image.open(image)
    array = np.array(image)
    lst.append(array)
MU = np.mean(lst)
STD = np.std(lst)    
# print('mean:', MU, 'std:', STD, 'min:', np.min(lst), 'max:', np.max(lst))
def normalize(im):
    """
    Normalize numpy array using MU and STD
    """
    im = np.array(im)
    im = (im - MU)/STD
    return im

def inv_normalize(im):
    """
    Inverse normalize numpy array using MU and STD
    Returns: Image
    """
    #im = im.numpy()
    im = np.squeeze(im)
    im = im * STD + MU
    im = im.astype('uint16')
    im = Image.fromarray(im, 'I;16')
    return im

class MyDataset(Dataset):
    def __init__(self, image_paths, target_paths, augment):
        self.image_paths = image_paths
        self.target_paths = target_paths
        self.augment = augment

    def transform(self, image, mask, augment):
        if augment:
            # Random rotate
            rotate = transforms.RandomRotation(180)
            angle = rotate.get_params(rotate.degrees)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

            # Random horizontal flipping
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

        image = normalize(image)
        image = TF.to_tensor(image)
        # mask = np.array(mask)
        mask = mask2onehot(mask)
        mask = torch.tensor(mask, dtype=torch.long)
        mask = torch.unsqueeze(mask,0)
        
        return image, mask

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        mask = Image.open(self.target_paths[index])
        x, y = self.transform(image, mask, augment=self.augment)
        return x, y

    def __len__(self):
        return len(self.image_paths)

# Instantiate Datasets
def create_loader(batch_size=64):
  train_dataset = MyDataset(imgs_train, msks_train, augment=True)
  val_dataset = MyDataset(imgs_val, msks_val, augment=False)

  BATCH_SIZE = batch_size
  dataloader_train = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
  dataloader_val = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
  return dataloader_train,dataloader_val
  # print('n_batches_train:', len(dataloader_train), 'n_batches_val:', len(dataloader_val))
