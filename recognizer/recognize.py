# Recognize module
# Author: Andreas Pentaliotis
# Module to implement handwriting recognition on the given input

import matplotlib.pyplot as plt

from PIL import Image
from load import load_data
from preprocess import binarize

data = load_data()

for i in data:
  plt.imshow(Image.fromarray(binarize(i)))
  plt.show()
