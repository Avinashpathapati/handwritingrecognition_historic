# Utility module
# Author: Andreas Pentaliotis
# Implementation of utility functions.

import cv2 as cv
import os
import random
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def plot(image, name):
  cv.imshow(name, image)
  cv.waitKey(0)
  cv.destroyWindow(name)

def read_image(path):
  return cv.imread(path, cv.IMREAD_UNCHANGED)

def load_data(path):
  print("loading data...")
  
  # Load the images and labels.
  images = []
  labels = []
  for directory in os.listdir(path + "/"):
    for filename in os.listdir(path + "/" + str(directory) + "/"):
      image = read_image(path + "/" + str(directory) + "/" + str(filename))
      images.append(image)
      labels.append(str(directory))

  return images, labels

def load_images(path):
  print("loading images...")

  # Load the images and their filenames.
  images = []
  filenames = []
  for filename in os.listdir(path + "/"):
    image = read_image(path + "/" + str(filename))
    images.append(image)
    filenames.append(str(filename))

  return images, filenames

def randomize(images, labels):
  print("shuffling data...")
  
  data = list(zip(images, labels))
  random.Random(1).shuffle(data)
  images[:], labels[:] = zip(*data)
  
  return images, labels

def split(images, labels):
  print("splitting data...")

  (x_train, x_test, y_train, y_test) = train_test_split(images, labels, test_size=0.25, random_state=1)
  
  return x_train, x_test, y_train, y_test

def parse_input_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--images", required=True, help="path to input character images")
  arguments = vars(parser.parse_args())

  if not os.path.isdir(arguments["images"]):
    raise Exception("path to input character images not found")
  elif not os.listdir(arguments["images"]):
    raise Exception("no files found in path to input character images")

  return arguments

def make_predictions(model, images, generator=None):
  print("making predictions...")

  if generator is None:
    predictions = model.predict(images)
  else:
    # Make the predictions by performing test time data augmentation using the given generator. For
    # this to make sense the model should have been trained with the same generator.
    predictions = []
    for _ in range(20):
      current_predictions = model.predict_generator(generator.flow(images, batch_size=1, shuffle=False), steps=images.shape[0])
      predictions.append(current_predictions)
    predictions = np.mean(predictions, axis=0)
  
  return predictions

def analyze(predictions, filenames):
  print("analyzing predictions...")

  # Determine the labels and probabilities and store them into a dataframe with
  # the image filenames.
  analyzed_predictions = pd.DataFrame()
  labels = ["Alef", "Ayin", "Bet", "Dalet", "Gimel", "He", "Het", "Kaf", "Kaf-final",
            "Lamed", "Mem", "Mem-medial", "Nun-final", "Nun-medial", "Pe", "Pe-final",
            "Qof", "Resh", "Samekh", "Shin", "Taw", "Tet", "Tsadi-final", "Tsadi-medial",
            "Waw", "Yod", "Zayin"]
  indices = np.argmax(predictions, axis=1)
  probabilities = np.max(predictions, axis=1)
  analyzed_predictions["names"] = filenames
  analyzed_predictions["labels"] = [labels[x] for x in indices]
  analyzed_predictions["probabilities"] = probabilities

  return analyzed_predictions

def save(analyzed_predictions):
  print("saving analyzed predictions to ./output...")

  # Save the analyzed predictions.
  if not os.path.isdir("./output"):
    os.mkdir("./output")
  analyzed_predictions.to_csv("./output/analyzed-predictions.csv")