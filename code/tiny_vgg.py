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

class TinyVGG(nn.Module):

  """
  Architecture replicates TinyVGG.

  CNN Model used to classify emotions from images of human faces.
  Input params:
  input_shape = shape of input (color channels)
  number_of_hidden_units = the number of hidden units for the model
  output_shape = number of classes that the model can predict
  """

  def __init__(self, input_shape: int, hidden_units: int ,output_shape: int) -> None:
    super().__init__()

    self.first_block = nn.Sequential(
      nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3,padding=2,stride = 1),
      nn.ReLU(),
      nn.Conv2d(in_channels = hidden_units, out_channels = hidden_units, kernel_size=3,padding=2,stride = 1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size= 3, padding = 1),
      nn.Dropout(0.25)
    )


    self.second_block = nn.Sequential(
        nn.Conv2d(in_channels = hidden_units, out_channels = hidden_units, kernel_size=3,padding=1,stride = 1),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Conv2d(in_channels = hidden_units, out_channels = hidden_units, kernel_size=3,padding=1,stride = 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size= 3, padding = 1),
        nn.Dropout(0.25)
        )

    self.third_block = nn.Sequential(
        nn.Conv2d(in_channels = hidden_units, out_channels = hidden_units, kernel_size=4,padding=1,stride = 1),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Conv2d(in_channels = hidden_units, out_channels = hidden_units, kernel_size=4,padding=1,stride = 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size= 3, padding = 1),
        nn.Dropout(0.25)
    )

    self.fourth_block = nn.Sequential(
        nn.Conv2d(in_channels = hidden_units, out_channels = hidden_units, kernel_size=3,padding=1,stride = 1),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Conv2d(in_channels = hidden_units, out_channels = hidden_units, kernel_size=3,padding=1,stride = 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size= 2, padding = 1)
    )

    self.classification_block = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=hidden_units*4*4,out_features = 128),
        nn.Dropout(0.25),
        nn.Linear(in_features=128,out_features = 64),
        nn.Linear(in_features=64,out_features = output_shape)
    )

  def forward(self, x):
    x  = self.first_block(x)
    x = self.second_block(x)
    #print(x.shape)
    x = self.third_block(x)
    #x = self.fourth_block(x)
    x = self.classification_block(x)
    #print(x.shape)
    return x