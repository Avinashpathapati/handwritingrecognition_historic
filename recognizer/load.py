# Load module
# Author: Andreas Pentaliotis
# Module to implement loading of the data

import argparse
import os
import cv2


def load_data():
  # Parse the input arguments.
  parser = argparse.ArgumentParser()
  parser.add_argument("-d", "--data", required=True,
                  help="path to input image data directory")
  #parser.add_argument("-p", "--plot", required=True,
                  #help="path to output images with recognized characters")
  arguments = vars(parser.parse_args())

  print("loading images...")
  data = []

  # Load the grayscale images into the data list.
  arguments["data"] = arguments["data"] + "/"
  for file in os.listdir(arguments["data"]):
    if "fused" in str(file):
      image = cv2.imread(arguments["data"] + str(file), cv2.IMREAD_GRAYSCALE)
      data.append(image)

  return data
