# Preprocessing module
# Author: Andreas Pentaliotis
# Module to implement preprocessing of the data

import cv2 as cv
import numpy as np

from matplotlib import pyplot as plt


def binarize(image):
  #image = cv.medianBlur(image,21)

  #image = cv.adaptiveThreshold(image,255,cv.ADAPTIVE_THRESH_MEAN_C,
                              #cv.THRESH_BINARY,11,2)
  #image = cv.adaptiveThreshold(image,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                              #cv.THRESH_BINARY,11,2)

  image = cv.GaussianBlur(image,(5,5),0)
  image = cv.threshold(image,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)[1]

  return image