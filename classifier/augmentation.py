# Augmentation module
# Module to implement data augmentation.

import random
import numpy as np
import cv2 as cv
import pandas as pd


def scale(image):
  (height, width) = image.shape[:2] 
  scale_type = random.choice([cv.INTER_CUBIC, cv.INTER_AREA])
  height = int(1.5 * height) if scale_type == cv.INTER_AREA else int(height / 1.5)
  width = int(1.5 * width) if scale_type == cv.INTER_AREA else int(width / 1.5)
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

def augment(data):
  scaled_data = pd.DataFrame()
  scaled_images = [scale(x) for x in data["images"]]
  scaled_data["images"] = scaled_images
  scaled_data["labels"] = data["labels"]

  rotated_data = pd.DataFrame()
  rotated_images = [rotate(x) for x in data["images"]]
  rotated_data["images"] = rotated_images
  rotated_data["labels"] = data["labels"]

  translated_data = pd.DataFrame()
  translated_images = [translate(x) for x in data["images"]]
  translated_data["images"] = translated_images
  translated_data["labels"] = data["labels"]

  data = data.append(scaled_data)
  data = data.append(rotated_data)
  data = data.append(translated_data)

  return data
