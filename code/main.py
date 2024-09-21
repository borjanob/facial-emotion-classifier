import cv2 as cv
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torchvision import datasets,transforms
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from ultralytics import YOLO
from pipeline import find_faces,get_faces_and_emotions,predict_emotion
from resnet import ResidualBlock,ResNet
from tiny_vgg import TinyVGG
# PATH TO MODELS
FACE_DETECTOR_ROOT = ''
PROBLEMATIC_EMOTIONS_CLASSIFIER = ''
EMOTIONS_CLASSIFIER = ''

data_transform = transforms.Compose([
     transforms.Grayscale(num_output_channels=1),
    transforms.Resize(size=(96, 96)),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

if __name__ == '__main__':
    
    photo_path = ''
    face_detector = YOLO(FACE_DETECTOR_ROOT)
    emotion_classifier = torch.load(f=EMOTIONS_CLASSIFIER, map_location='cpu')
    problematic_emotions_classifier = torch.load(f=PROBLEMATIC_EMOTIONS_CLASSIFIER,  map_location='cpu')
    emotions = original_folder_names = ['anger','contempt','happy','neutral','problematic','sad']
    problematic_emotions = ['disgust', 'fear', 'surprise']
    labels = get_faces_and_emotions(photo_path,face_detector,emotion_classifier, problematic_emotions_classifier,data_transform,emotions,problematic_emotions)