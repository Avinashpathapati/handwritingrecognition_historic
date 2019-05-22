# CNN module
# Author: Andreas Pentaliotis
# Convolutional neural network implementation.

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.constraints import maxnorm
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
K.set_image_dim_ordering("tf")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import numpy as np


class CNN():

  def __init__(self, height, width, depth, classes):
    self.height = height
    self.width = width
    self.depth = depth
    self.classes = classes
    self.__build()

  def __build(self):
    input_shape = (self.height, self.width, self.depth)

    # Build the model and compile it.
    self.model = Sequential()
  
    self.model.add(Conv2D(128, (5, 5), input_shape=input_shape, padding="same"))
    self.model.add(Activation("relu"))
    self.model.add(MaxPooling2D(pool_size=(2, 2)))
  
    self.model.add(Conv2D(64, (5, 5), padding="same"))
    self.model.add(Activation("relu"))
    self.model.add(MaxPooling2D(pool_size=(2, 2)))

    self.model.add(Flatten())
    self.model.add(Dense(1024))
    self.model.add(Activation("relu"))
    self.model.add(Dropout(0.2))

    self.model.add(Dense(self.classes, activation="softmax"))

    self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

  def summary(self):
    print()
    print()
    print("CNN")
    print("--------------------")
    self.model.summary()

  def train(self, x_train, y_train, epochs, batch_size, generator=None, validation_split=0):
    # Handle any value errors on the input arguments.
    if batch_size > x_train.shape[0]:
      raise ValueError("batch size should be less than size of data")
    if batch_size <= 0:
      raise ValueError("batch size should be greater than 0")
    if not isinstance(batch_size, int):
      raise ValueError("batch size should be an integer")
    
    if epochs <= 0:
      raise ValueError("epochs should be greater than 0")
    if not isinstance(epochs, int):
      raise ValueError("epochs should be an integer")

    if validation_split < 0:
      raise ValueError("validation split should be greater than 0 or equal to 0")
    if validation_split >= 1:
      raise ValueError("validation split should be less than 1")
      
    if generator is not None and validation_split != 0:
      # Train using the given generator for data augmentation, with validation split.
      generator._validation_split = validation_split
      training_generator = generator.flow(x_train, y_train, batch_size=batch_size, subset="training")
      validation_generator = training_generator = generator.flow(x_train, y_train, batch_size=batch_size, subset="validation")
      
      fitting = self.model.fit_generator(training_generator, validation_data=validation_generator,
                                         steps_per_epoch=int(x_train.shape[0] / batch_size),
                                         validation_steps=int(x_train.shape[0] * validation_split / batch_size),
                                         epochs=epochs)
    elif generator is not None:
      # Train using the given generator for data augmentation, without validation split.
      training_generator = generator.flow(x_train, y_train, batch_size=batch_size)
      
      fitting = self.model.fit_generator(training_generator, steps_per_epoch=int(x_train.shape[0] / batch_size), epochs=epochs)
    elif validation_split != 0:
      fitting = self.model.fit(x_train, y_train, validation_split=validation_split, epochs=epochs, batch_size=batch_size)
    else:
      fitting = self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    # Save the training results.
    if not os.path.isdir("./output"):
      os.mkdir("./output")

    if validation_split != 0:
      plt.plot(fitting.history["acc"])
      plt.plot(fitting.history["val_acc"])
      plt.title("CNN accuracy")
      plt.ylabel("Accuracy")
      plt.xlabel("Epoch")
      plt.legend(["Training set", "Validation set"], loc="upper left")
      plt.savefig("./output/cnn-fitting-accuracy")
      plt.close()

      plt.plot(fitting.history["loss"])
      plt.plot(fitting.history["val_loss"])
      plt.title("CNN loss")
      plt.ylabel("Loss")
      plt.xlabel("Epoch")
      plt.legend(["Training set", "Validation set"], loc="upper left")
      plt.savefig("./output/cnn-fitting-loss")
      plt.close()
    else:
      plt.plot(fitting.history["acc"])
      plt.title("CNN training accuracy")
      plt.ylabel("Accuracy")
      plt.xlabel("Epoch")
      plt.savefig("./output/cnn-fitting-accuracy")
      plt.close()

      plt.plot(fitting.history["loss"])
      plt.title("CNN training loss")
      plt.ylabel("Loss")
      plt.xlabel("Epoch")
      plt.savefig("./output/cnn-fitting-loss")
      plt.close()

  def test(self, x_test, y_test, generator=None):
    if generator is not None:
      # Make the evaluation by performing test time data augmentation using the given generator. For
      # this to make sense the model should have been trained with the same generator.
      evaluation = []
      for _ in range(20):
        current_evaluation = self.model.evaluate_generator(generator.flow(x_test, y_test, batch_size=1, shuffle=False),
                                                           steps=x_test.shape[0], verbose=1)
        evaluation.append(current_evaluation)
      evaluation = np.mean(evaluation, axis=0)
    else:
      evaluation = self.model.evaluate(x_test, y_test)

    if not os.path.isdir("./output"):
      os.mkdir("./output")

    with open("./output/cnn-evaluation.txt", "w+") as output_file:
      output_file.write("Model's loss on unseen data: " + str(evaluation[0]))
      output_file.write("\nModel's accuracy on unseen data: " + str(evaluation[1]))

  def save(self):
    if not os.path.isdir("./output"):
      os.mkdir("./output")
    self.model.save("./output/cnn.h5")
