# Preprocessing module
# Author: Andreas Pentaliotis
# Module to implement preprocessing of the data

import cv2 as cv
import numpy as np

from matplotlib import pyplot as plt


def normalize(image):
  return image / 255

def smooth(image):
  #image = cv.medianBlur(image, 5)
  image = cv.GaussianBlur(image, (5, 5), 0)
  return image

def binarize(image):
  #image = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                              #cv.THRESH_BINARY, 5, 2)
  #image = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                               #cv.THRESH_BINARY, 3, 1)
  image = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
  return image

def preprocess(data):
  print("preprocessing images...")
  
  data = [smooth(x) for x in data]
  data = [binarize(x) for x in data]
  data = [normalize(x) for x in data]

  return data
