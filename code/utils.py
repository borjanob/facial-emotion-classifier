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


def walk_throuhh_directory(directory_path):
  for dirpath,dirnames,filenames in os.walk(directory_path):
    print(f'There are {len(dirnames)} directories and {len(filenames)} files in {dirpath}')

def unzip_folder(folder_path):
  with zipfile.ZipFile(folder_path, 'r') as zip:
    zip.extractall()
    print('Done')



def divide_training_data(original_folder_names: list,labels: pd.DataFrame,splitting_limit_percentage:int = 80, home_path: str = '/content', labels_provided: bool = True, train_extension: str = '/train', test_extension: str = '/test'):

  """

  Divides images in {original_folder_names} folders into training and testing data with correct labels.
  Labels are provided in csv form either in labels variable or are the names of the folders.

  """

  train_path = home_path + train_extension
  test_path = home_path + test_extension

  if os.path.exists(train_path) == False:
      os.mkdir(train_path)
  if os.path.exists(test_path) == False:
      os.mkdir(test_path)

  image_counter = 0
  error_counter = 0
  for folder_name in original_folder_names:

    if os.path.exists(train_path + '/' + folder_name) == False:
      os.mkdir(train_path + '/' + folder_name)

    if os.path.exists(test_path + '/' + folder_name) == False:
      os.mkdir(test_path + '/' + folder_name)

    files_in_folder = os.listdir('/content/' + folder_name)
    number_of_files_in_folder = len(files_in_folder)

    limit = int(number_of_files_in_folder / 100 *80)

    for index, file_name in enumerate(files_in_folder):

      full_path_of_image = folder_name + '/' + file_name

      #get true label from labels dataset
      try:
        label_of_image = labels.loc[labels['pth'] == full_path_of_image].label.item()

        img = Image.open('/content/' + full_path_of_image)
        if index <= limit:
          img.save( train_path + '/'+ label_of_image + '/image' + str(image_counter) + file_name[-4:])
        else:
          img.save(test_path + '/'+ label_of_image + '/image' + str(image_counter) + file_name[-4:])

        image_counter +=1
      except:
        error_counter +=1

  print(f'Saved: {image_counter} images, with {error_counter} errors')


def divide_training_data_with_problematic_labels(original_folder_names: list,problematic_labels:list ,labels: pd.DataFrame,splitting_limit_percentage:int = 80, home_path: str = '/content', labels_provided: bool = True):

  """
  Divides images in {original_folder_names} into folders with names and problematic images provided in {problematic_labels}
  """

  train_path = home_path + '/train_with_problematic'
  test_path = home_path + '/test_with_problematic'

  if os.path.exists(train_path) == False:
      os.mkdir(train_path)
  if os.path.exists(test_path) == False:
      os.mkdir(test_path)

  regular_image_counter = 0
  train_problematic_image_counter = 0
  problematic_image_counter = 0
  error_counter = 0
  for folder_name in original_folder_names:
    problematic = False

    if folder_name in problematic_labels:
      problematic = True

      if os.path.exists(train_path + '/problematic') == False:
        os.mkdir(train_path + '/problematic')

      if os.path.exists(test_path + '/problematic') == False:
        os.mkdir(test_path + '/problematic')



    if not problematic:

      if os.path.exists(train_path + '/' + folder_name) == False:
        os.mkdir(train_path + '/' + folder_name)

      if os.path.exists(test_path + '/' + folder_name) == False:
        os.mkdir(test_path + '/' + folder_name)

    files_in_folder = os.listdir('/content/' + folder_name)
    number_of_files_in_folder = len(files_in_folder)

    limit = int(number_of_files_in_folder / 100 * splitting_limit_percentage)

    for index, file_name in enumerate(files_in_folder):
        full_path_of_image = folder_name + '/' + file_name

        #get true label from labels dataset
        try:
            label_of_image = labels.loc[labels['pth'] == full_path_of_image].label.item()

            img = Image.open('/content/' + full_path_of_image)
            if index <= limit:
              if problematic:
                img.save(train_path + '/problematic/image' + str(regular_image_counter) + file_name[-4:])
                problematic_image_counter+=1
                train_problematic_image_counter+=1
                #regular_image_counter+=1
              else:
                img.save(train_path + '/' + label_of_image + '/image' + str(regular_image_counter) + file_name[-4:])
                #regular_image_counter +=1
            else:
              if problematic:
                img.save(test_path+ '/problematic/image' + str(regular_image_counter) + file_name[-4:])
                problematic_image_counter+=1
                #regular_image_counter+=1
              else:
                img.save(test_path + '/' + label_of_image + '/image' + str(regular_image_counter) + file_name[-4:])
            regular_image_counter +=1
        except:
            error_counter +=1

  print(f'Saved: {regular_image_counter} regular images,{train_problematic_image_counter} train problematic images,{problematic_image_counter} all problematic images, with {error_counter} errors')



def count_files_in_folder(folder_path):
  return len([name for name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, name))])