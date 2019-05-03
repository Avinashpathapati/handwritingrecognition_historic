# Utility module
# Utility function for character classifier.

import cv2 as cv
import pandas as pd
import os


def plot(image, name):
  cv.imshow(name, image)
  cv.waitKey(0)
  cv.destroyWindow(name)

def read_image(path):
  return cv.imread(path, cv.IMREAD_UNCHANGED)

def pad(image, width, height):
  top = (height - image.shape[0]) // 2
  bottom = (height - image.shape[0]) // 2
  left = (width - image.shape[1]) // 2
  right = (width - image.shape[1]) // 2

  # Correct any rounding errors in divisions.
  if image.shape[1] + left + right != width:
    left += 1
  if image.shape[0] + top + bottom != height:
    top += 1

  return cv.copyMakeBorder(image, top, bottom, left, right, cv.BORDER_CONSTANT, value=255)

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
