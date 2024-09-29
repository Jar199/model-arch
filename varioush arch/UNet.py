import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.double_conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
  def forward(self, x):
    return self.double_conv(x)


class Downsampl(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.downsampl = nn.Sequential(
        nn.MaxPool2d(2),
        DoubleConv(in_channels, out_channels)
    )
  def forward(self, x):
    return self.downsampl(x)


class Upsampl(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.upsampl = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    self.conv = DoubleConv(in_channels, out_channels)

  def forward(self, x2, x1):
    x2 = self.upsampl(x2)
    # B, C, H, W
    diffY = x1.size()[2] - x2.size()[2]
    diffX = x1.size()[3] - x2.size()[3]
    x2 = F.pad(x2, [(diffX + 1)//2, diffX//2, (diffY + 1//2), diffY//2])
    x = torch.cat([x1, x2], dim=1)
    x = self.conv(x)
    return x

class Unet(nn.Module):
  def __init__(self, in_channels, num_classes):
    super().__init__()
    self.inconv = DoubleConv(in_channels, 64)
    self.down1 = Downsampl(64, 128)
    self.down2 = Downsampl(128, 256)
    self.down3 = Downsampl(256, 512)
    self.maxpool = nn.MaxPool2d(2)
    self.conv1_1 = nn.Conv2d(512, 1024, kernel_size=1)
    self.conv1_2 = nn.Conv2d(1024, 1024, kernel_size=1)

    self.up1 = Upsampl(1024, 512)
    self.up2 = Upsampl(512, 256)
    self.up3 = Upsampl(256, 128)
    self.up4 = Upsampl(128, 64)
    self.outconv = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=1)

  def forward(self, x):
    x1 = self.inconv(x)
    x2 = self.down1(x1)
    x3 = self.down2(x2)
    x4 = self.down3(x3)
    x = self.maxpool(x4)
    x = self.conv1_1(x)
    x = self.conv1_2(x)

    x = self.up1(x, x4)
    x = self.up2(x, x3)
    x = self.up3(x, x2)
    x = self.up4(x, x1)
    x = self.outconv(x)
    return x
