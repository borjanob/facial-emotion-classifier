import torch
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
from ultralytics import YOLO

def find_faces(img_path: str,face_detector: YOLO, threshold: int = 0.5) -> list:

  """

  Returns location of faces detected on image with path {img_path}

  """

  img = cv.imread(img_path)

  predictions = face_detector(img_path)[0]
  faces = []
  face_locations = dict()
  for index,prediction in enumerate(predictions.boxes.data.tolist()):
    x1,y1,x2,y2,score,class_id = prediction
    if score > threshold:
      face = img[int(y1):int(y2),int(x1):int(x2)]
      faces.append(face)
      face_locations[img_path + str(index)] = [x1,x2,y1,y2]

  return faces, face_locations


def predict_emotion(emotion_classifier: nn.Module, problematic_emotion_classifier: nn.Module,
                    face_image: Image, transforms: torchvision.transforms,
                    emotion_labels: list, problematic_emotions_labels: list,
                    device:str = 'cpu' ) -> str:

  transformed = transforms(face_image)
  problematic = False
  #label = ''
  emotion_classifier.eval()
  with torch.inference_mode():
    predictions = emotion_classifier(transformed.unsqueeze(dim=1)).to(device)
  predicted_class = predictions.argmax(dim=1)

  label = emotion_labels[predicted_class]

  if predicted_class == 4:
    problematic = True
    problematic_emotion_classifier.eval()
    with torch.inference_mode():
      preds = problematic_emotion_classifier(transformed.unsqueeze(dim=1)).to(device)
    predicted_class = preds.argmax(dim=1)

  if problematic:
    #                                !!!!!!
    label = problematic_emotions_labels[0]

  return label


def get_faces_and_emotions(img_path: str, face_detector: YOLO,
                    emotion_classifier: nn.Module, problematic_emotion_classifier: nn.Module,
                     transforms: torchvision.transforms,
                    emotion_labels: list, problematic_emotions_labels: list,
                    threshold:int = 0.5,
                    device:str = 'cpu' ):

  image = cv.imread(img_path)
  faces,face_locations = find_faces(img_path, face_detector)
  labels = []
  for index,face in enumerate(faces):
    face = Image.fromarray(face)
    emotion = predict_emotion(emotion_classifier, problematic_emotion_classifier, face, transforms, emotion_labels, problematic_emotions_labels)
    corners = face_locations[img_path + str(index)]
    x1,x2,y1,y2 = corners[0], corners[1], corners[2], corners[3]
    labels.append(emotion)
    cv.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
    cv.putText(image, emotion,(int(x1) + 40 ,int(y1)), cv.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv.LINE_AA)

  plt.imshow(image)
  plt.show()
  return labels
