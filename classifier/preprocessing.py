# Preprocessing module
# Author: Andreas Pentaliotis
# Implementation of preprocessing functions.

import cv2 as cv
import numpy as np
from random import shuffle
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split


def __pad(image, width, height):
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

def __binarize(image):
  return cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]

def __randomize(images, labels):
  data = list(zip(images, labels))
  shuffle(data)
  images[:], labels[:] = zip(*data)
  return images, labels

def preprocess_training(images, labels):
  print("preprocessing data...")

  images, labels = __randomize(images, labels)
  
  # One hot encode the labels.
  binarizer = LabelBinarizer()
  labels = binarizer.fit_transform(labels)

  # Pad all the images with white pixels to maximum height and maximum width.
  max_width = np.amax(np.unique([x.shape[1] for x in images]))
  max_height = np.amax(np.unique([x.shape[0] for x in images]))
  images = [__pad(x, max_width, max_height) for x in images]

  images = [__binarize(x) for x in images]
  images = [x / 255 for x in images]

  # Split the data into training and testing and reshape the image arrays.
  (x_train, x_test, y_train, y_test) = train_test_split(images, labels, test_size=0.25, random_state=1)
  x_train = np.array(x_train, dtype=np.uint8)
  x_test = np.array(x_test, dtype=np.uint8)
  x_train = np.reshape(x_train, (x_train.shape[0], max_height, max_width, 1))
  x_test = np.reshape(x_test, (x_test.shape[0], max_height, max_width, 1))
  
  return x_train, x_test, y_train, y_test

def preprocess_testing(images):
  print("preprocessing images...")

  # Pad all the images with white pixels to maximum height and maximum width.
  max_width = np.amax(np.unique([x.shape[1] for x in images]))
  max_height = np.amax(np.unique([x.shape[0] for x in images]))
  images = [__pad(x, max_width, max_height) for x in images]

  images = [__binarize(x) for x in images]
  images = [x / 255 for x in images]

  return images
