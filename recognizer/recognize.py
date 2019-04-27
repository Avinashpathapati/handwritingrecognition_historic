# Recognize module
# Author: Andreas Pentaliotis
# Module to implement handwriting recognition on the given input

from load import load_data
from utility import plot, plot_histogram
from preprocess import binarize, smooth, normalize, preprocess


data = load_data()

#for image in data:
  #plot_histogram(image)

data = preprocess(data)
for image in data:
  plot(image)