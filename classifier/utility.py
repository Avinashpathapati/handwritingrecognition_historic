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

def resize(image):
  return cv.resize(image, (256, 256))

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
