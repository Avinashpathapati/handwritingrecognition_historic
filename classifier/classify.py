# Classify module
# Author: Andreas Pentaliotis
# Module to implement character classification.

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

from utility import load_images, parse_input_arguments, make_predictions, analyze, save
from preprocessing import preprocess

arguments = parse_input_arguments()

# Load and preprocess the images.
images, filenames = load_images(arguments["images"])
images = preprocess(images)

# Load the model and make the predictions. Use a generator with the same settings as in
# training to make the predictions better.
cnn = load_model("/home/anpenta/Desktop/character-classifier/handwritingrecognition/cnn-data-augmentation/cnn.h5")
generator = ImageDataGenerator(zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1, rotation_range=5)
predictions = make_predictions(cnn, images, generator=generator)

analyzed_predictions = analyze(predictions, filenames)
print(analyzed_predictions["labels"].value_counts()) # For inspection - to be deleted.
save(analyzed_predictions)
