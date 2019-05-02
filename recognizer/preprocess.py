# Preprocessing module
# Author: Andreas Pentaliotis
# Module to implement functions for preprocessing the images.

import cv2 as cv
import numpy as np
from recognizer.utility import load_single_image,plot_opencv

from matplotlib import pyplot as plt


def normalize(image):
  return image / 255

def smooth(image):
  #image = cv.medianBlur(image, 5)
  image = cv.GaussianBlur(image, (5, 5), 8)
  return image

def binarize(image):
  #image = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                              #cv.THRESH_BINARY, 5, 2)
  #image = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                               #cv.THRESH_BINARY, 3, 1)
  image = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
  return image

def thresholded_binarisation(image,threshold):
  image = cv.threshold(image, threshold, 255,cv.THRESH_BINARY)[1]
  return image


def preprocess(data):
  print("preprocessing images...")
  
  data = [smooth(x) for x in data]
  data = [binarize(x) for x in data]
  data = [normalize(x) for x in data]

  return data


def preprocess_single():
  image_path = '../data/test/'
  image_name = '0_test.jpg'

  img = load_single_image(image_path, image_name,load_greyscale=True)
  img = smooth(img)
  img = binarize(img)
  img = normalize(img)

  print(np.unique(img))
  plot_opencv(img)

  cv.imwrite("../data/test/test_binary.png", img)
