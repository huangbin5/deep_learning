import numpy as np
import imageio
import cv2
from PIL import Image
import torch

model = torch.hub.load('ultralytics/yolov3', 'yolov3')
img = '1.jpg'
result = model(img)
result.save()
