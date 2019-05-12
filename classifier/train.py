# Train module
# Author: Andreas Pentaliotis
# Module to implement training of a model on the character data.

import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from utility import load_data, randomize, split
from preprocessing import preprocess_training
from cnn import CNN


images, labels = load_data("../character-data")
classes = len(np.unique(labels))

images, labels = preprocess_training(images, labels)
images, labels = randomize(images, labels)
x_train, x_test, y_train, y_test = split(images, labels)

# Delete the images and labels arrays to free memory.
del images
del labels

cnn = CNN(x_train.shape[1], x_train.shape[2], x_train.shape[3], classes)
cnn.summary()

generator = ImageDataGenerator(zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1, rotation_range=5)

cnn.train(x_train, y_train, epochs=30, batch_size=32, generator=generator)
cnn.test(x_test, y_test, generator=generator)
cnn.save()
