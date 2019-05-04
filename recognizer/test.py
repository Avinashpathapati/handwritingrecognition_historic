# Test module
# Module to test the other modules.

from recognizer.utility import plot_histogram, load_data, get_input_arguments
from recognizer.preprocess import binarize, smooth, normalize, preprocess


arguments = get_input_arguments()
data = load_data(arguments["data"])

#for image in data:
  #plot_histogram(image)

#"""

# [Temporary] Saving the images to inspect.
data = preprocess(data)
import cv2 as cv
import os
import matplotlib.pyplot as plt
if not os.path.exists("../output"):
  os.makedirs("../output")
i = 0
for image in data:
  plt.imsave(os.path.join('../output/', str(i) + '.jpg'), image)
  i += 1
#"""