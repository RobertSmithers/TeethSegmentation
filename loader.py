import os
import random
from re import X
import numpy as np
import pandas as pd
import cv2
from PIL import Image

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from torch.utils.data  import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch.nn as nn
import torch.nn.functional as F


###Pre-Process Images###
def convert_one_channel(img):
    #some images have 3 channels , although they are grayscale image
    if len(img.shape)>2:
        img=img[:,:,0]
        return img
    else:
        return img

def pre_images(resize_shape,path):
    img=Image.open(path)
    img=img.resize((resize_shape),Image.ANTIALIAS)
    img=convert_one_channel(np.asarray(img))
    cv2.imwrite(path,img)
    return img

def show_imgs():
    fig = plt.figure(figsize = (30,7))
    for index in range(7):
      file_path1 = os.path.join('original_img', str(index+1)+'.png')
      file_path2 = os.path.join('masked_img', str(index+1)+'.png')
      # print(file_path)
      ax = fig.add_subplot(2, 7, index+1)
      plt.imshow(pre_images((512,512),file_path1))  #show result of converting every img to one color channel

      ax = fig.add_subplot(2, 7, index+8)
      plt.imshow(cv2.imread(file_path2))


def gen_csv():
    ###generate file names to keep track of file for later usage
    arr1=np.arange(1,117)
    # print(arr1.dtype)
    arr1=arr1.astype(str)
    # print(type(arr1))
    df=pd.DataFrame(arr1)
    df[1]=df[0]
    df[0]=df[0]+'.png'
    df.to_csv('data/sample.csv',index=False)
###Load DATA

# def rename():
#     ###rename all files in masked_img folder
#     folder='data/masked_img'
#     for file_name in os.listdir(folder):
#         source = folder+'/'+file_name
#         destination = source.replace('_m.png','.png')
#         os.rename(source, destination)

# def mergefiles():
#     ###move renamed files to Images folder
#     folder='data/masked_img'
#     for file_name in os.listdir(folder):
#       source = folder+'/'+file_name
#       destination = 'data/original_img/'+file_name
#       os.rename(source,destination)

def get_fnames(root):
  xs, ys = os.listdir(os.path.join(root, 'original_img')), os.listdir(os.path.join(root, 'masked_img'))
  f = lambda fname: int(fname.split('.png')[0])
  xs = sorted(xs, key=f)
#   ys = sorted(ys, key=f)
  return xs

# our dataset class
rest_set_size = 0.3
test_set_size = 0.5
class dset(Dataset):
    def __init__(self, data, root_dir='data', train=False, transformX = None, transformY = None):

      self.root_dir = root_dir
      self.transformX = transformX
      self.transformY = transformY
      self.train = train
      self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        fname = self.data[index]

        imx_name = os.path.join(self.root_dir, 'original_img', fname)
        imy_name = os.path.join(self.root_dir, 'masked_img', fname)
        imx = Image.open(imx_name)
        imy = Image.open(imy_name).convert('L')

        ##data augmentation
        if self.train:
          #Random horizontal flipping
          if random.random() > 0.5:
              imx = TF.hflip(imx)
              imy = TF.hflip(imy)

          #Random vertical flipping
          if random.random() > 0.5:
              imx = TF.vflip(imx)
              imy = TF.vflip(imy)

          #Random rotation
          if random.random() > 0.8:
            angle = random.choice([-30, -90, -60, -45 -15, 0, 15, 30, 45, 60, 90])
            imx = TF.rotate(imx, angle)
            imy = TF.rotate(imy, angle)

        if self.transformX:
            imx = self.transformX(imx)
            imy = self.transformY(imy)

        sample = {'image': imx, 'annotation': imy}
        return sample

tx_X = transforms.Compose([ transforms.Resize((3100, 1200)),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,))
                              ])
tx_Y = transforms.Compose([ transforms.Resize((3100, 1200)),
                              transforms.ToTensor()
                              ])

all_data = get_fnames(root='data')

# split the dataset to train and rest
# split the rest to validation and test
train_data, other_data = train_test_split(all_data, test_size = rest_set_size, random_state = 5)
val_data, test_data = train_test_split(other_data, test_size = test_set_size, random_state = 5)

train_set = dset(train_data, 'data', train = True, transformX = tx_X, transformY = tx_Y)
val_set = dset(val_data, 'data', transformX = tx_X, transformY = tx_Y)
test_set = dset(test_data, 'data', transformX = tx_X, transformY = tx_Y)

train_loader = DataLoader(dataset=train_set, batch_size=2, shuffle=True, num_workers=1)
val_loader = DataLoader(dataset=val_set, batch_size=2, shuffle=True, num_workers=1)
test_loader = DataLoader(dataset=test_set, batch_size=2, shuffle=True, num_workers=1)
