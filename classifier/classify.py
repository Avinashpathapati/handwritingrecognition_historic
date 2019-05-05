# Classify module
# Author: Andreas Pentaliotis
# Module to implement character classification.

from keras.models import load_model
import numpy as np
import argparse
import os
import pandas as pd

from utility import load_images
from preprocessing import preprocess_testing


# Parse the input arguments.
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--images", required=True, help="path to input character images")
arguments = vars(parser.parse_args())

if not os.path.isdir(arguments["images"]):
  raise Exception("path to input character images not found")
elif not os.path.isfile(arguments["images"] + "/*"):
  raise Exception("no files found in path to input character images")

images, names = load_images(arguments["images"])
images = preprocess_testing(images)

# Load the model and make the predictions.
print("making predictions...")
cnn = load_model("cnn.h5")
predictions = cnn.predict(images)

# Determine the labels and probabilities and store them into a dataframe with
# the image names.
data = pd.DataFrame()
labels = ["Alef", "Ayin", "Bet", "Dalet", "Gimel", "He", "Het", "Kaf", "Kaf-final"
          "Lamed", "Mem", "Mem-medial", "Nun-final", "Nun-medial", "Pe", "Pe-final"
          "Qof", "Resh", "Samekh", "Shin", "Taw", "Tet", "Tsadi-final", "Tsadi-medial"
          "Waw", "Yod", "Zayin"]
indices = np.argmax(predictions, axis=1)
probabilities = np.max(predictions, axis=1)
data["names"] = names
data["labels"] = [labels[x] for x in indices]
data["probabilities"] = probabilities

# Save the data.
print("saving predictions to ./output...")
if not os.path.isdir("./output"):
  os.mkdir("./output")
data.to_csv("./output/predictions.csv")
