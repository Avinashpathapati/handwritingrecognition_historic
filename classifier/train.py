# Train module
# Module to implement training of a model on the character data.

import numpy as np
import matplotlib.pyplot as plt

from utility import load_data
from preprocessing import preprocess
from augmentation import augment
from cnn import build_cnn


images, labels = load_data("/home/anpenta/Desktop/character-classifier/data/monkbrill-jpg/monkbrill2")
classes = len(np.unique(labels))

images, labels = augment(images, labels)
x_train, x_test, y_train, y_test = preprocess(images, labels)

# Delete the original lists to free memory.
del images[:]
del labels[:]

model = build_cnn(x_train.shape[1], x_train.shape[2], 1, classes)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
print(model.summary())

history = model.fit(x_train, y_train, validation_split=0.25, epochs=500, batch_size=32)
model.save("model.h5")

plt.plot(history.history["acc"])
plt.plot(history.history["val_acc"])
plt.title("Model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Training set", "Validation set"], loc="upper left")
plt.savefig("model-fit-accuracy")
plt.show()

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Training set", "Validation set"], loc="upper left")
plt.savefig("model-fit-loss")
plt.show()
