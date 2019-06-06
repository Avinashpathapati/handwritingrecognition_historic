# Preprocessing module
# Module to implement functions for preprocessing the images.

import cv2 as cv
import numpy as np
from recognizer.utility import load_single_image,plot_opencv, plot_matplotlib
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola)
from recognizer.utility import invert_image

from matplotlib import pyplot as plt
from external_code import sauvola


def normalize(image):
  return image / 255

def smooth(image):
  image = cv.medianBlur(image, 3)
  #image = cv.GaussianBlur(image, (3, 3), 0)
  return image

def binarize(image):
  #image = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_MEAN_C,
  #cv.THRESH_BINARY, 11, 2)
  #image = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
  #cv.THRESH_BINARY, 11, 2)
  image = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
  return image


def binarize_sauvola(image):
  imbw = sauvola.binarize(image, [75, 75],64, 0.3)
  return imbw

def thresholded_binarisation(image,threshold):
  image = cv.threshold(image, threshold, 255,cv.THRESH_BINARY)[1]
  return image


def bilateral_filter_noise_removal(img):
  image = cv.bilateralFilter(img,10,5,5)
  return image

def whiten_background(image,mask):
  kernel_size = 10
  kernel = np.ones((kernel_size, kernel_size), np.uint8)
  mask = cv.erode(mask, kernel)
  #plot_opencv(mask)

  image[mask != 255] = 255
  return image



def post_process(image,mask):
  image[mask == 255] = 255
  return image


def additional_binarisation(image):
  window_size = 25
  thresh_niblack = threshold_niblack(image, window_size=window_size, k=0.8)
  #thresh_sauvola = threshold_sauvola(image, window_size=window_size)

  binary_niblack = image > thresh_niblack
  binary_niblack = 255. * binary_niblack
  #binary_sauvola = image > thresh_sauvola
  #binary_sauvola = 255. * binary_sauvola
  return binary_niblack


def niblack_and_otsu_binarisation(image,window_size=5):
  window_size=window_size
  thresh_niblack = threshold_niblack(image, window_size=window_size, k=0.8)
  binary_niblack = image > thresh_niblack
  binary_niblack = 255 * binary_niblack

  image_otsu = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]

  final_image = np.ones(image.shape,dtype=np.uint8)
  final_image[binary_niblack == 0] = 0
  final_image[image_otsu == 0] = 0

  final_image = 255 * final_image

  return final_image


def niblack_and_otsu_edge_binarisation(image,window_size=5):
  window_size=window_size
  thresh_niblack = threshold_niblack(image, window_size=window_size, k=0.8)
  binary_niblack = image > thresh_niblack
  binary_niblack = 255 * binary_niblack

  image_otsu = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]

  final_image = np.ones(image.shape,dtype=np.uint8)
  final_image[binary_niblack == 0] = 0
  final_image[image_otsu == 0] = 0

  final_image = 255 * final_image
  #plot_opencv(final_image)

  edges = edge_detection(final_image)
  final_image = post_process(final_image,edges)
  # final_image = np.ones(image.shape, dtype=np.uint8)
  # final_image[]

  #plot_opencv(edges)
  return final_image


def otsu_edge_binarisation(image):

  image_otsu = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
  edges = edge_detection(image_otsu)
  final_image = post_process(image_otsu,edges)
  # final_image = np.ones(image.shape, dtype=np.uint8)
  # final_image[]

  #plot_opencv(edges)
  return final_image


def remove_border(image):
  kernel_size = 5
  kernel = np.ones((kernel_size,kernel_size),np.uint8)
  mask = cv.erode(image, kernel)
  mask = cv.erode(image, kernel)
  mask = cv.erode(image, kernel)
  mask = cv.bitwise_not(mask)
  #plot_matplotlib(mask)
  image = cv.bitwise_and(image, image, mask = mask)
  image = cv.bitwise_not(image)
  return image

def enhance(image):
  kernel_size = 3
  kernel = np.ones((kernel_size,kernel_size),np.uint8)
  image = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)
  return image


def edge_detection(img):
  edges = cv.Canny(img,100,200)

  return edges


def enhance_2(im):
  kernel = np.ones((3, 3), np.uint8)
  im = cv.erode(im, kernel, iterations=1)
  kernel = np.ones((3, 3), np.uint8)
  im = cv.dilate(im, kernel, iterations=1)

  return im


def area_closing(img):
  kernel=np.ones((3, 3), np.uint8)
  closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
  return closing


def area_opening(img):
  kernel = np.ones((3, 3), np.uint8)
  opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
  return opening


def conv2blurring(image):
  kernel = np.ones((3, 3), np.float32) / 25
  dst = cv.filter2D(image, -1, kernel)

  return dst

def preprocess(data):
  print("preprocessing images...")

  images = [x[0] for x in data]
  masks = [x[1] for x in data]
  images = [smooth(x) for x in images]
  images = [conv2blurring(x) for x in images]
  images = [otsu_edge_binarisation(x) for x in images]
  images = [whiten_background(x, y) for x, y in zip(images, masks)]

  ####################
  images = [abs(255-area_opening(abs(255-x))) for x in images]
  images = [abs(255-area_closing(abs(255-x))) for x in images]

  images = [abs(255 - area_opening(abs(255 - x))) for x in images]
  images = [abs(255 - area_closing(abs(255 - x))) for x in images]

  images = [abs(255 - area_opening(abs(255 - x))) for x in images]
  images = [abs(255 - area_closing(abs(255 - x))) for x in images]
  ######################

  return images

def preprocess_single():
  image_path = '../data/test/'
  image_name = '0_test.jpg'

  img = load_single_image(image_path, image_name,load_greyscale=True)
  #img = smooth(img)
  #img = binarize(img)
  #img = normalize(img)

  img=additional_binarisation(img)
  print(np.unique(img))
  plot_matplotlib(img)

  #cv.imwrite("../data/test/test_binary.png", img)
