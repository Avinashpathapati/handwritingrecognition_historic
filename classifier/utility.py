# Utility module
# Author: Andreas Pentaliotis
# Implementation of utility functions.

import cv2 as cv
import os
import random


def plot(image, name):
  cv.imshow(name, image)
  cv.waitKey(0)
  cv.destroyWindow(name)

def read_image(path):
  return cv.imread(path, cv.IMREAD_UNCHANGED)

def load_data(path):
  print("loading data...")
  
  # Load the images and labels.
  images = []
  labels = []
  for directory in os.listdir(path + "/"):
    for filename in os.listdir(path + "/" + str(directory) + "/"):
      image = read_image(path + "/" + str(directory) + "/" + str(filename))
      images.append(image)
      labels.append(str(directory))

  return images, labels

def load_images(path):
  print("loading images...")

  # Load the images and their filenames.
  images = []
  names = []
  for directory in os.listdir(path + "/"):
    for filename in os.listdir(path + "/" + str(directory) + "/"):
      image = read_image(path + "/" + str(directory) + "/" + str(filename))
      images.append(image)
      names.append(str(filename))

  return images, names

def randomize(images, labels):
  print("shuffling data...")
  
  data = list(zip(images, labels))
  random.Random(1).shuffle(data)
  images[:], labels[:] = zip(*data)
  
  return images, labels