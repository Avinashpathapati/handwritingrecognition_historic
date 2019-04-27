# Utility module
# Author: Andreas Pentaliotis
# Module to implement utility functions

import matplotlib.pyplot as plt

from PIL import Image

def plot(image):
  plt.imshow(Image.fromarray(image))
  plt.show()

def plot_histogram(image):
  plt.hist(image.ravel(), bins=256, range=[0,256])
  plt.xlabel("Pixel value")
  plt.ylabel("Number of pixels")
  plt.show()
