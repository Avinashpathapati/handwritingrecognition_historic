# Preprocessing module
# Author: Andreas Pentaliotis
# Implementation of preprocessing functions.

import cv2 as cv
import numpy as np
from sklearn.preprocessing import LabelBinarizer


def __resize(images, dimensions):
  images = [cv.resize(x, dimensions) for x in images]
  return images

def __binarize(images):
  images = [cv.threshold(x, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1] for x in images]
  images = [x / 255 for x in images]
  return images

def preprocess(images, labels=None):
  print("preprocessing data...")

  # Preprocess the images.
  images = __resize(images, dimensions=(64, 64))
  images = __binarize(images)
  images = np.array(images)
  images = np.reshape(images, (images.shape[0], images.shape[1], images.shape[2], 1))

  if labels is not None:
    # One hot encode the labels.
    binarizer = LabelBinarizer()
    labels = binarizer.fit_transform(labels)
    return images, labels

  return images
