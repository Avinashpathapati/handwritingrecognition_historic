# CNN module
# Author: Andreas Pentaliotis
# Convolutional neural network implementation.

from keras.preprocessing.image import ImageDataGenerator
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

  def train(self, x_train, y_train, epochs, batch_size, augment_data=None):
    if augment_data:
      generator = ImageDataGenerator(zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1, rotation_range=5,
                               validation_split=0.25)
      training_generator = generator.flow(x_train, y_train, batch_size=batch_size, subset="training")
      validation_generator = training_generator = generator.flow(x_train, y_train, batch_size=batch_size, subset="validation")
      
      fitting = self.model.fit_generator(training_generator, validation_data=validation_generator,
                                         steps_per_epoch=int(x_train.shape[0] / batch_size),
                                         validation_steps = int(x_train.shape[0] * 0.25 / batch_size),
                                         epochs=epochs)
    else:
      fitting = self.model.fit(x_train, y_train, validation_split=0.25, epochs=epochs, batch_size=batch_size)

    if not os.path.isdir("./output"):
      os.mkdir("./output")

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

  def test(self, x_test, y_test):
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
