# Train module
# Module to implement training of a model on the character data.

import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from utility import load_data, pad, plot, binarize
from augmentation import augment
from cnn import build_cnn


# Load the data, augment them, shuffle them and get the number of classes.
images, labels = load_data("/home/anpenta/Desktop/character-classifier/data/monkbrill-jpg/monkbrill2")
images, labels = augment(images, labels)
data = list(zip(images, labels))
shuffle(data)
images[:], labels[:] = zip(*data)
classes = len(np.unique(labels))

# Pad all the images with white pixels to maximum height and maximum width.
max_width = np.amax(np.unique([x.shape[1] for x in images]))
max_height = np.amax(np.unique([x.shape[0] for x in images]))
images = [pad(x, max_width, max_height) for x in images]

# Binarize the characters to match the test data and normalize the pixel values.
images = [binarize(x) for x in images]
images = [x / 255 for x in images]

# Split the data into training and testing and reshape the image arrays.
(x_train, x_test, y_train, y_test) = train_test_split(images, labels, test_size=0.25, random_state=1)
x_train = np.array(x_train, dtype=np.uint8)
x_test = np.array(x_test, dtype=np.uint8)
x_train = np.reshape(x_train, (x_train.shape[0], max_height, max_width, 1))
x_test = np.reshape(x_test, (x_test.shape[0], max_height, max_width, 1))

# One hot encode the labels.
binarizer = LabelBinarizer()
y_train = binarizer.fit_transform(y_train)
y_test = binarizer.transform(y_test)

# Build and compile the model and show the summary.
model = build_cnn(max_height, max_width, 1, classes)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
print(model.summary())

# Train the model using a validation split on the training data and save the model and
# the resulting plots
history = model.fit(x_train, y_train, validation_split=0.25, epochs=1, batch_size=16)
model.save("model.h5")

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper left')
plt.savefig("model-fit-accuracy")
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper left')
plt.savefig("model-fit-loss")
plt.show()
