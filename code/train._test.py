from tqdm.auto import tqdm
import cv2 as cv
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import torchvision
from torchvision import datasets,transforms
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pandas as pd
import torch.optim.lr_scheduler as lr_scheduler
import zipfile
from pathlib import Path
import pandas as pd


def train_model(model: torch.nn.Module, train_dataloader: torch.utils.data.DataLoader,
                loss_function: torch.nn.Module, optimizer: torch.optim.Optimizer ,device: str = 'cpu') -> int:


  """
  Defines one step of model training

  return avg. epoch loss
  """

  epoch_loss = 0

  model.to(device)

  model.train()

  for batch, (X,y) in enumerate(train_dataloader):

    X = X.to(device)
    Y = y.to(device)

    predictions = model(X)

    loss = loss_function(predictions,Y)

    epoch_loss += loss

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()


  epoch_loss /= len(train_dataloader)
  print(f'Avg. loss for epoch: {epoch_loss}')
  #print(model.state_dict())
  return epoch_loss


def test_model(model: torch.nn.Module, test_dataloader: torch.utils.data.DataLoader,
                loss_function: torch.nn.Module, metric = None,device: str = 'cpu'):

  """
  Performs one step of testing on model

  returns: avg. test_loss, avg. evaluation metric
  """

  model.to(device)

  test_loss,metric_loss = 0,0

  model.eval()
  with torch.inference_mode():

    for X,Y in test_dataloader:

      X = X.to(device)
      Y = Y.to(device)

      predictions = model(X)

      loss = loss_function(predictions,Y)

      test_loss += loss

      # if metric is not None:
      #   metric = metric(y_pred = predictions.cpu().argmax(dim=1), y_true = Y.cpu())
      #   metric_loss += metric

    # if metric is not None:
    #   metric_loss /= len(test_dataloader)
    test_loss /= len(test_dataloader)

  print(f'Avg. test loss: {test_loss}, avg. evaluation metric: {metric_loss if metric is not None else 0}')

  return test_loss,metric_loss


def train_for_epochs(model: torch.nn.Module, train_dataloader: torch.utils.data.DataLoader,
                     test_dataloader: torch.utils.data.DataLoader,
                      loss_function: torch.nn.Module,
                     optimizer: torch.optim.Optimizer,
                     scheduler,
                     epochs: int = 3, metric = None,device: str = 'cpu'):

  """
  Performs model training and testing for multiple epochs
  """
  for epoch in tqdm(range(epochs)):
    print(f'Epoch: {epoch}\n')
    train_model(model,train_dataloader,loss_function,optimizer,device)
    test_model(model,test_dataloader,loss_function,metric,device)
    print(f'Lr: {optimizer.param_groups[0]["lr"]}')
    #scheduler.step()

