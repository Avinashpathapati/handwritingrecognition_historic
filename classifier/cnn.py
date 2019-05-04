# CNN module
# Convolutional neural network implementation.

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.constraints import maxnorm
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers.core import BatchNormalization
from keras import backend as K


def build_cnn(height, width, depth, classes):
  # Determine the input shape.
  if K.image_data_format() == "channels_first":
    input_shape = (depth, height, width)
  else:
    input_shape = (height, width, depth)

  # Build the model
  model = Sequential()
  
  model.add(Conv2D(32, (11, 11), input_shape=input_shape, padding="same"))
  model.add(BatchNormalization())
  model.add(Activation("relu"))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  
  model.add(Conv2D(64, (7, 7), padding="same"))
  model.add(BatchNormalization())
  model.add(Activation("relu"))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(128, (5, 5), padding="same"))
  model.add(BatchNormalization())
  model.add(Activation("relu"))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  
  model.add(Conv2D(256, (3, 3), padding="same"))
  model.add(BatchNormalization())
  model.add(Activation("relu"))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Flatten())
  model.add(Dense(512))
  model.add(BatchNormalization())
  model.add(Activation("relu"))
  model.add(Dropout(0.5))

  model.add(Dense(classes, activation="softmax"))

  return model
