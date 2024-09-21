import torch
from torch import nn
import torchvision
from torchvision import datasets,transforms
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pandas as pd
import torch.optim.lr_scheduler as lr_scheduler


class ResidualBlock(nn.Module):
  def __init__(self, in_channels, hidden_units, stride = 1, downsample = None) -> None:
    super().__init__()

    self.first_conv_block = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=hidden_units, kernel_size = 3, stride = stride, padding = 1),
        nn.Dropout(0.25),
        nn.BatchNorm2d(hidden_units),
        nn.ReLU()
    )

    self.second_conv_block = nn.Sequential(
        nn.Conv2d(in_channels=hidden_units, out_channels = hidden_units, kernel_size = 3, padding =1 ,stride = 1 ),
        nn.BatchNorm2d(hidden_units)
    )

    self.relu = nn.ReLU()

    self.downsample = downsample

  def forward(self, x : torch.Tensor) -> torch.Tensor:

    # make data copy
    input = x

    x = self.first_conv_block(x)
    x = self.second_conv_block(x)

    if self.downsample:

      # transform residual if activ. function is added
      input = self.downsample(input)

    # add residual to conv blocks

    x+=input

    x = self.relu(x)
    return x
  



class ResNet(nn.Module):

  def __init__(self,input_shape: int,block_type : nn.Module, layer_sizes: list, num_of_classes: int = 9, inplanes: int = 64 ) -> None:
    super().__init__()

    self.inplanes = inplanes

    self.first_conv = nn.Sequential(
        nn.Conv2d(in_channels=input_shape,out_channels = inplanes, kernel_size= 7, stride = 2),
        nn.Dropout(0.25),
        nn.BatchNorm2d(inplanes),
        nn.ReLU()
    )

    self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
    self.avgpool = nn.AvgPool2d(2, stride=1)

    self.layer1 = self._build_block(block_type, inplanes, layer_sizes[0], False, stride =1)
    self.layer2 = self._build_block(block_type, 128, layer_sizes[1], True, stride=2)
    self.layer3 = self._build_block(block_type, 256, layer_sizes[2], True, stride = 2)
    self.layer4 = self._build_block(block_type, 512, layer_sizes[3], True, stride = 2)

    self.output_layer =nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=512,out_features = num_of_classes),
    )

    self.dropout1 = nn.Dropout(0.25)
    self.dropout2 = nn.Dropout(0.5)

  def _build_block(self,block_type: nn.Module, out_planes:int, num_of_blocks:int,contain_downsample:bool = False, stride:int =1) -> nn.Module:

    downsample = None

    # if residual should go through function
    if stride != 1 or self.inplanes != out_planes:
      downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, out_planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_planes),
            )

    # layers of residual block will be added here
    layers = []

    # first layer with input size = to output of previous layer
    layers.append(block_type(self.inplanes, out_planes, stride, downsample))
    # change to output
    self.inplanes = out_planes

    # add remaining layers
    for i in range(1,num_of_blocks):
      layers.append(block_type(self.inplanes,out_planes))

    return nn.Sequential(*layers)


  def forward(self,x):

    x = self.first_conv(x)
    x = self.maxpool(x)
   #print(x)
    x = self.layer1(x)
    x = self.dropout1(x)
    x = self.layer2(x)
    x = self.dropout1(x)
    x = self.layer3(x)
    x = self.dropout1(x)
    x = self.layer4(x)
    #x = self.dropout1(x)
    x = self.avgpool(x)

    #x = x.view(x.size(0), -1)
    x = self.output_layer(x)
    return x