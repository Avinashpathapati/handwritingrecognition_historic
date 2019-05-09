# Utility module
# Author: Andreas Pentaliotis
# Module to implement utility functions for recognizer.

import matplotlib.pyplot as plt
import argparse
import os
import cv2 as cv

from PIL import Image


def plot(image):
  plt.imshow(Image.fromarray(image))
  plt.show()

def plot_histogram(image):
  plt.hist(image.ravel(), bins=256, range=[0,256])
  plt.xlabel("Pixel value")
  plt.ylabel("Number of pixels")
  plt.show()

def get_input_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument("-d", "--data", required=True,
                      help="path to input image data directory")
  arguments = vars(parser.parse_args())

  return arguments

def load_data(path, search_str):
  print("loading images...")
  data = []

  # Load the grayscale images into the data list.
  path = path + "/"
  print(path)
  for file in os.listdir(path):
    print(file)
    if search_str in str(file):
      print(search_str)
      image = cv.imread(path + str(file), cv.IMREAD_GRAYSCALE)
      data.append(image)

  return data