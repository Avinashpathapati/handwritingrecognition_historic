# Splitting module
# Author: Andreas Pentaliotis
# Implementation of splitting functions.

import numpy as np
from random import shuffle
from sklearn.model_selection import train_test_split


def __randomize(images, labels):
  data = list(zip(images, labels))
  shuffle(data)
  images[:], labels[:] = zip(*data)
  return images, labels

def split(images, labels, classifier):
  print("splitting data randomly...")

  images, labels = __randomize(images, labels)

  # Split the data into training and testing.
  (x_train, x_test, y_train, y_test) = train_test_split(images, labels, test_size=0.25, random_state=1)
  x_train = np.array(x_train, dtype=np.uint8)
  x_test = np.array(x_test, dtype=np.uint8)

  # Reshape the image arrays depending on the classifier.
  if classifier == "cnn":
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))
  elif classifier == "stacked_lstm":
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))
  else:
    raise ValueError("classifier not recognized")
  
  return x_train, x_test, y_train, y_test