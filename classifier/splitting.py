# Splitting module
# Author: Andreas Pentaliotis
# Implementation of splitting functions.

import numpy as np
import random
from sklearn.model_selection import train_test_split


def split(images, labels):
  print("splitting data...")

  # Split the data into training and testing.
  (x_train, x_test, y_train, y_test) = train_test_split(images, labels, test_size=0.25, random_state=1)
  x_train = np.array(x_train)
  x_test = np.array(x_test)

  # Reshape the image arrays.
  x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
  x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))
  
  return x_train, x_test, y_train, y_test
