# Utility module
# Utility function for character classifier.

import cv2 as cv
import pandas as pd
import os
import random
import numpy as np


def plot(image, name):
  cv.imshow(name, image)
  cv.waitKey(0)
  cv.destroyWindow(name)

def read_image(path):
  return cv.imread(path, cv.IMREAD_UNCHANGED)

def scale(image):
  (height, width) = image.shape[:2] 
  scale_type = random.choice([cv.INTER_CUBIC, cv.INTER_AREA])
  height = int(1.5 * height) if scale_type == cv.INTER_AREA else int(height / 1.5)
  width = int(1.5 * width) if scale_type == cv.INTER_AREA else int(width / 1.5)
  print(scale_type)
  return cv.resize(image, (width, height), interpolation = scale_type) 

def rotate(image):
  (rows, columns) = image.shape[:2]
  rotation_degree = random.choice([-10, -5, 5, 10])
  matrix = cv.getRotationMatrix2D((columns / 2, rows / 2), rotation_degree, 1) 
  return cv.warpAffine(image, matrix, (columns, rows), borderValue = 255) 

def translate(image):
  (rows, columns) = image.shape[:2] 
  horizontal_shift = random.choice([-4, -3, -2, -1, 0, 1, 2, 3, 4])
  vertical_shift = random.choice([-4, -3, -2, -1, 0, 1, 2, 3, 4])
  matrix = np.float32([[1, 0, horizontal_shift], [0, 1, vertical_shift]]) 
  return cv.warpAffine(image, matrix, (columns, rows), borderValue = 255)

def load_data(path):
  print("loading images...")
  
  # Load the images and labels into a dataframe.
  data = pd.DataFrame()
  images = []
  labels = []
  for directory in os.listdir(path + "/"):
    for filename in os.listdir(path + "/" + str(directory) + "/"):
        image = read_image(path + "/" + str(directory) + "/" + str(filename))
        images.append(image)
        labels.append(str(directory))

  data["images"] = images
  data["labels"] = labels
  
  return data
