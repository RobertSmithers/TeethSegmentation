import os
import numpy as np
import cv2
from PIL import Image

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from torch.utils.data  import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch.nn as nn
import torch.nn.functional as F

# Extra augmentation (blur, noise, brightness)
# import albumentations as A


###Pre-Process Images###
# def convert_one_channel(img):
#     #some images have 3 channels , although they are grayscale image
#     if len(img.shape)>2:
#         img=img[:,:,0]
#         return img
#     else:
#         return img

# def pre_images(resize_shape,path):
#     img=Image.open(path)
#     img=img.resize((resize_shape),Image.ANTIALIAS)
#     img=convert_one_channel(np.asarray(img))
#     cv2.imwrite(path,img)
#     return img

# def show_imgs():
#     fig = plt.figure(figsize = (30,7))
#     for index in range(7):
#       file_path1 = os.path.join('original_img', str(index+1)+'.png')
#       file_path2 = os.path.join('masked_img', str(index+1)+'.png')
#       # print(file_path)
#       ax = fig.add_subplot(2, 7, index+1)
#       plt.imshow(pre_images((512,512),file_path1))  #show result of converting every img to one color channel

#       ax = fig.add_subplot(2, 7, index+8)
#       plt.imshow(cv2.imread(file_path2))


# def gen_csv():
#     ###generate file names to keep track of file for later usage
#     arr1=np.arange(1,117)
#     # print(arr1.dtype)
#     arr1=arr1.astype(str)
#     # print(type(arr1))
#     df=pd.DataFrame(arr1)
#     df[1]=df[0]
#     df[0]=df[0]+'.png'
#     df.to_csv('data/sample.csv',index=False)
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
    def __init__(self, data, root_dir='data', train=False, transformX = None, transformY = None, batch_transforms = 3):

      self.root_dir = root_dir
      self.transformX = transformX
      self.transformY = transformY
      self.train = train
      self.data = data
      self.batch_transforms = batch_transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        fname = self.data[index]

        imx_name = os.path.join(self.root_dir, 'original_img', fname)
        imy_name = os.path.join(self.root_dir, 'masked_img', fname)
        imx = Image.open(imx_name)
        imy = Image.open(imy_name).convert('L')
        
        # 3104 and 1200 need to be divisible by 16 to work in our model.
        imx=imx.resize((512,512), Image.ANTIALIAS)
        imy=imy.resize((512,512), Image.ANTIALIAS)

        ##data augmentation
#         if self.train:
            
#           for _ in range(self.batch_transforms):
#             # Set of augmentations shown to be successful in SerdarHelli's usage in X-Ray imagery
#             aug = A.Compose([
#                     A.OneOf([A.RandomCrop(height=512, width=512),
#                                 A.PadIfNeeded(min_height=512, min_width=512, p=0.5)],p=0.4),
#                     A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25,p=0.5),
#                     A.Compose([A.RandomScale(scale_limit=(-0.15, 0.15), p=1, interpolation=1),
#                                             A.PadIfNeeded(min_height=512, min_width=512, border_mode=cv2.BORDER_CONSTANT), 
#                                             A.Resize(512, 512, cv2.INTER_NEAREST), ],p=0.5),
#                     A.ShiftScaleRotate (shift_limit=0.325, scale_limit=0.15, rotate_limit=15,border_mode=cv2.BORDER_CONSTANT, p=1),
#                     A.Rotate(15,p=0.5),
#                     A.Blur(blur_limit=1, p=0.5),
#                     A.Downscale(scale_min=0.15, scale_max=0.25,  always_apply=False, p=0.5),
#                     A.GaussNoise(var_limit=(0.05, 0.1), mean=0, per_channel=True, always_apply=False, p=0.5),
#                     A.HorizontalFlip(p=0.25),
#             ])
            
#             augd = aug(image=np.array(imx), mask=np.array(imy))
#             imx, imy = augd['image'], augd['mask']
            
#             del augd
#             del aug

        if self.transformX:
            imx = self.transformX(imx)
            imy = self.transformY(imy)

        sample = {'image': imx, 'annotation': imy}
        return sample

tx_X = transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,))
                              ])
tx_Y = transforms.Compose([
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
val_loader = DataLoader(dataset=val_set, batch_size=1, shuffle=True, num_workers=1)
test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=True, num_workers=1)
