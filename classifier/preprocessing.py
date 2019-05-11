# Preprocessing module
# Author: Andreas Pentaliotis
# Implementation of preprocessing functions.

import cv2 as cv
import numpy as np
from sklearn.preprocessing import LabelBinarizer


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

def preprocess_training(images, labels):
  print("preprocessing data...")
  
  # One hot encode the labels.
  binarizer = LabelBinarizer()
  labels = binarizer.fit_transform(labels)

  # Pad all the images with white pixels to maximum height and maximum width.
  max_width = np.amax(np.unique([x.shape[1] for x in images]))
  max_height = np.amax(np.unique([x.shape[0] for x in images]))
  images = [__pad(x, max_width, max_height) for x in images]

  images = [__binarize(x) for x in images]
  images = [x / 255 for x in images]

  return images, labels

def preprocess_testing(images):
  print("preprocessing images...")

  # Pad all the images with white pixels to maximum height and maximum width.
  max_width = np.amax(np.unique([x.shape[1] for x in images]))
  max_height = np.amax(np.unique([x.shape[0] for x in images]))
  images = [__pad(x, max_width, max_height) for x in images]

  images = [__binarize(x) for x in images]
  images = [x / 255 for x in images]

  return images
