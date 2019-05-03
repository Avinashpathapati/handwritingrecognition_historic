# Utility module
# Module to implement utility functions for recognizer.

import matplotlib.pyplot as plt
import argparse
import os
import cv2 as cv
import numpy as np

from PIL import Image


def plot_opencv(image):
  winname='PLOT'
  cv.imshow(winname,image)
  cv.waitKey(0)
  cv.destroyWindow(winname)

def plot_matplotlib(image):
  plt.imshow(Image.fromarray(image))
  plt.show()

def plot_histogram(image):
  plt.hist(image.ravel(), bins=256, range=[0,256])
  plt.xlabel("Pixel value")
  plt.ylabel("Number of pixels")
  plt.show()

def save_opencv(image, path, name):
  cv.imwrite(os.path.join(path, name), image)

def get_input_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument("-d", "--data", required=True,
                      help="path to input image data directory")
  arguments = vars(parser.parse_args())

  return arguments

def load_data(path):
  print("loading images...")
  data = []

  # Load the grayscale images into the data list.
  path = path + "/"
  for file in os.listdir(path):
    if "fused" in str(file):
      image = cv.imread(path + str(file), cv.IMREAD_GRAYSCALE)
      data.append(image)

  return data

def load_single_image(image_path,image_name,load_greyscale=False):
  img = image_path + str(image_name)
  if not os.path.exists(img):
    raise Exception('Path does not exist')

  if load_greyscale:
    image = cv.imread(image_path + str(image_name),cv.IMREAD_GRAYSCALE)
  else:
    print('in here')
    image = cv.imread(image_path + str(image_name),cv.IMREAD_UNCHANGED)

  #image = cv.imread(image_path + str(image_name), cv.IMREAD_GRAYSCALE)
  return image