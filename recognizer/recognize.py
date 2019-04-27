# Recognize module
# Author: Andreas Pentaliotis
# Module to implement handwriting recognition on the given input

import matplotlib.pyplot as plt

from PIL import Image
from load import load_data
from preprocess import binarize, smooth, normalize, preprocess


data = load_data()

"""
process = smooth
for i in data:
  plt.imshow(Image.fromarray(preprocess(i)))
  plt.show()
"""

#"""
data = preprocess(data)
for i in data:
  plt.imshow(Image.fromarray(i))
  plt.show()
#"""
