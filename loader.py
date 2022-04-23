import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from torch.utils.data  import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

###Pre-Process DATA###

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
      
###Load DATA

# our dataset class
rest_set_size = 0.3
test_set_size = 0.5
class dset(Dataset):
    def __init__(self, root_dir='data', train=True,test=True,transformX = None, transformY = None):
      
      try:
        self.pixel_file = pd.read_csv(os.path.join(root_dir, 'sample.csv'))
      except:
        df = pd.DataFrame(np.arange(1,117))
        df.to_csv(os.path.join(root_dir,'sample.csv'))
        self.pixel_file = pd.read_csv(os.path.join(root_dir, 'sample.csv'))
        
      self.root_dir = root_dir
      self.transformX = transformX
      self.transformY = transformY
      self.train = train
      self.test = test

      # split the dataset to train and rest
      # split the rest to validation and test
      self.train_data, self.rest_data = train_test_split(self.pixel_file, test_size = rest_set_size, random_state = 5)
      self.validation_data, self.test_data = train_test_split(self.rest_data, test_size = test_set_size, random_state = 5)

    def __len__(self):
        if self.train:
          length = len(self.train_data)
        elif self.test:
          length = len(self.test_data)
        else:
          length = len(self.validation_data)
        return length
    
    def __getitem__(self, index):
        if self.train:
          imx_name = os.path.join(self.root_dir, 'original_img', f'{self.train_data.iloc[index, 0]}.png')
          imy_name = os.path.join(self.root_dir, 'masked_img', f'{self.train_data.iloc[index, 0]}.png')
        elif self.test:
          imx_name = os.path.join(self.root_dir, 'original_img', f'{self.test_data.iloc[index, 0]}.png')
          imy_name = os.path.join(self.root_dir, 'masked_img', f'{self.test_data.iloc[index, 0]}.png')
        else:
          imx_name = os.path.join(self.root_dir, 'original_img', f'{self.validation_data.iloc[index, 0]}.png')
          imy_name = os.path.join(self.root_dir, 'masked_img', f'{self.validation_data.iloc[index, 0]}.png')
        
        imx = Image.open(imx_name)
        imy = Image.open(imy_name).convert('L')

        if self.transformX:
            imx = self.transformX(imx)
            imy = self.transformY(imy)
      
        sample = {'image': imx, 'annotation': imy}
        return sample
    
tx_X = transforms.Compose([ transforms.Resize((512, 512)),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,))
                              ])
tx_Y = transforms.Compose([ transforms.Resize((512, 512)),
                              transforms.ToTensor()
                              ])
train_data = dset('data', train = True, test=False, transformX = tx_X, transformY = tx_Y)
validation_data = dset('data', train = False,test=False,transformX = tx_X, transformY = tx_Y)
test_data = dset('data', train = False, test=True,transformX = tx_X, transformY = tx_Y)

train_loader = DataLoader(dataset=train_data, batch_size=2, shuffle=True, num_workers=2)
validation_loader = DataLoader(dataset=validation_data, batch_size=2, shuffle=True, num_workers=2)
test_loader = DataLoader(dataset=test_data, batch_size=2, shuffle=True, num_workers=2)