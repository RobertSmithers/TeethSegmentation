import torch
import torch.nn as nn
import torch.nn.functional as F

# NOTE: On torch version 1.7 (which I was previously on), the "same" padding option was not valid for Conv2d. Requires torch v1.10 and up

class DoubleConvBlock(nn.Module):
  # (conv) => (dropout+ReLU) => (conv) => (bn+ReLU)

  def __init__(self, in_ch, out_ch, kernel_size, batch_norm=True):
    super(DoubleConvBlock, self).__init__()

    if batch_norm:
      self.conv = nn.Sequential(
          nn.Conv2d(in_ch, out_ch, 3, padding="same"),
          nn.Dropout2d(0.1),
          nn.BatchNorm2d(out_ch),
          # nn.LeakyReLU(inplace=True),
          nn.Conv2d(out_ch, out_ch, 3, padding="same"),
          nn.BatchNorm2d(out_ch),             # Heavily debated, but have decided to normalize before activation
          nn.LeakyReLU(inplace=True)
      )
    else:
      self.conv = nn.Sequential(
          nn.Conv2d(in_ch, out_ch, 3, padding="same"),
          nn.Dropout2d(0.1),
          nn.LeakyReLU(inplace=True),
          nn.Conv2d(out_ch, out_ch, 3, padding="same"),
          nn.LeakyReLU(inplace=True)
      )

  def forward(self, x):
      return self.conv(x)

# This is our encoder
class PoolConvBlock(nn.Module):
  # (Pool) => (DoubleConvBlock)

  def __init__(self, pool_kernel_size=(2,2), conv_in_ch=32, conv_out_ch=32, conv_kernel_size=(3,3), batch_norm=True):
    super(PoolConvBlock, self).__init__()
    self.pool = nn.MaxPool2d(pool_kernel_size)
    self.conv = DoubleConvBlock(conv_in_ch, conv_out_ch, conv_kernel_size, batch_norm=batch_norm)

  def forward(self, x):
    p = self.pool(x)
    x = self.conv(p)
    return x, p

class TConv(nn.Module):
  # (ConvTranspose) => (ReLU)

  def __init__(self, in_ch, out_ch, kernel_size, stride=None):
    super(TConv, self).__init__()
    self.conv = nn.Sequential(
          nn.ConvTranspose2d(in_ch, out_ch, kernel_size, padding=1, stride=stride),
          # nn.LeakyReLU(inplace=True)
      )
  
  def forward(self, x, skip):
    x = self.conv(x)
    x = torch.cat([x, skip], axis=1)
    return x

class UNet(nn.Module):
    def __init__(self, input_shape=(512,512,1)):
        super(UNet, self).__init__()
        
        # conv1 = Conv2D(32,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        # d1=Dropout(0.1)(conv1)
        # conv2 = Conv2D(32,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(d1)
        # b=BatchNormalization()(conv2)
        self.conv = DoubleConvBlock(1, 32, (3,3))
        
        # pool1 = MaxPooling2D(pool_size=(2, 2))(b)
        # conv3 = Conv2D(64,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        # d2=Dropout(0.2)(conv3)
        # conv4 = Conv2D(64,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(d2)
        # b1=BatchNormalization()(conv4)
        self.pconv1 = PoolConvBlock(conv_in_ch=32, conv_out_ch=64)

        # pool2 = MaxPooling2D(pool_size=(2, 2))(b1)
        # conv5 = Conv2D(128,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        # d3=Dropout(0.3)(conv5)
        # conv6 = Conv2D(128,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(d3)
        # b2=BatchNormalization()(conv6)
        self.pconv2 = PoolConvBlock(conv_in_ch=64, conv_out_ch=128)
        
        # pool3 = MaxPooling2D(pool_size=(2, 2))(b2)
        # conv7 = Conv2D(256,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        # d4=Dropout(0.4)(conv7)
        # conv8 = Conv2D(256,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(d4)
        # b3=BatchNormalization()(conv8)
        self.pconv3 = PoolConvBlock(conv_in_ch=128, conv_out_ch=256)
        
        # pool4 = MaxPooling2D(pool_size=(2, 2))(b3)
        # conv9 = Conv2D(512,(3,3),activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        # d5=Dropout(0.5)(conv9)
        # conv10 = Conv2D(512,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(d5)
        # b4=BatchNormalization()(conv10)
        self.pconv4 = PoolConvBlock(conv_in_ch=256, conv_out_ch=512)
        
        # conv11 = Conv2DTranspose(512,(4,4), activation = 'relu', padding = 'same', strides=(2,2),kernel_initializer = 'he_normal')(b4)
        self.tconv1 = TConv(512, 256, (4,4), stride=(2,2))
        # x= concatenate([conv11,conv8])

        # conv12 = Conv2D(256,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
        # d6=Dropout(0.4)(conv12)
        # conv13 = Conv2D(256,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(d6)
        # b5=BatchNormalization()(conv13)
        self.conv1 = DoubleConvBlock(512, 256, (3,3))
        
        # conv14 = Conv2DTranspose(256,(4,4), activation = 'relu', padding = 'same', strides=(2,2),kernel_initializer = 'he_normal')(b5)
        self.tconv2 = TConv(256, 128, (4,4), stride=(2,2))
        # x1=concatenate([conv14,conv6])

        # conv15 = Conv2D(128,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x1)
        # d7=Dropout(0.3)(conv15)
        # conv16 = Conv2D(128,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(d7)
        # b6=BatchNormalization()(conv16)
        self.conv2 = DoubleConvBlock(256, 128, 3)
        
        # conv17 = Conv2DTranspose(128,(4,4), activation = 'relu', padding = 'same',strides=(2,2), kernel_initializer = 'he_normal')(b6)
        self.tconv3 = TConv(128, 64, (4,4), stride=(2,2))
        
        # x2=concatenate([conv17,conv4])

        # conv18 = Conv2D(64,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x2)
        # d8=Dropout(0.2)(conv18)
        # conv19 = Conv2D(64,(3,3) ,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(d8)
        # b7=BatchNormalization()(conv19)
        self.conv3 = DoubleConvBlock(128, 64, 3)

        # conv20 = Conv2DTranspose(64,(4,4), activation = 'relu', padding = 'same',strides=(2,2), kernel_initializer = 'he_normal')(b7)
        self.tconv4 = TConv(64, 32, (4,4), stride=(2,2))
        
        # x3=concatenate([conv20,conv2])

        # conv21 = Conv2D(32,(3,3) ,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x3)
        # d9=Dropout(0.1)(conv21)
        # conv22 = Conv2D(32,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(d9)
        self.conv4 = DoubleConvBlock(64, 32, (3,3), batch_norm=False)

        self.conv5 = nn.Sequential(
            nn.Conv2d(32, 1, (1,1), padding="same"),
            nn.ReLU()
        )
        # outputs = Conv2D(1,(1,1), activation = last_activation, padding = 'same', kernel_initializer = 'he_normal')(conv22)

    def forward(self, x):

      ### Reduce ###
      
      # Outputs 512 x 512 x 32
      skip1 = self.conv(x)
      # print("level 1 layer+skip", skip1.shape)

      # Outputs 256 x 256 x 64
      skip2, layer2 = self.pconv1(skip1)
      # print("level 2 layer/skip", layer2.shape, skip2.shape)

      # Outputs 128 x 128 x 128
      skip3, layer3 = self.pconv2(skip2)
      # print("level 3 layer/skip", layer3.shape, skip3.shape)

      # Outputs 64 x 64 x 256
      skip4, layer4 = self.pconv3(skip3)
      # print("level 4 layer/skip", layer4.shape, skip4.shape)
      
      # Outputs 32 x 32 x 512
      skip5, x = self.pconv4(skip4)
      # print("level 5 channels", x.shape)

      ### Expand ###
      
      # Outputs 64 x 64 x 256
      expand4 = self.tconv1(skip5, skip4)
      expand4 = self.conv1(expand4)

      # Outputs 128 x 128 x 128
      expand3 = self.tconv2(expand4, skip3)
      expand3 = self.conv2(expand3)

      # Outputs 256 x 256 x 64
      expand2 = self.tconv3(expand3, skip2)
      expand2 = self.conv3(expand2)

      # Outputs 512 x 512 x 32
      expand1 = self.tconv4(expand2, skip1)
      expand1 = self.conv4(expand1)

      # Outputs 512 x 512 x 1
      out = self.conv5(expand1)

      # return torch.sigmoid(x)
      # No sigmoid because using BCEWithLogistLoss which is more numerically stable (using log-sum-exp trick)
      return out