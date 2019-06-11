# Classify module
# Author: Andreas Pentaliotis
# Module to implement character classification.

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

from utility import load_images, parse_input_arguments, make_predictions, analyze, save
from preprocessing import preprocess


scrolls_path = "/home/anpenta/Desktop/character-transcription/handwritingrecognition/test"

cnn = load_model("/home/anpenta/Desktop/character-classifier/handwritingrecognition/cnn-data-augmentation/final-model/cnn.h5")
generator = ImageDataGenerator(zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1, rotation_range=5)
for scroll_folder in os.listdir(scrolls_path):
  for line_directory in os.listdir(scrolls_path + "/" + scroll_folder + "/"):
    for word_directory in os.listdir(scrolls_path + "/" + scroll_folder + "/" + line_directory + "/"):
      word_path = scrolls_path + "/" + scroll_folder + "/" + line_directory + "/" + word_directory + "/"

      # Load and preprocess the images.
      images, filenames = load_images(word_path)
      images = preprocess(images)
      
      predictions = make_predictions(cnn, images, generator=generator)
      
      analyzed_predictions = analyze(predictions, filenames)
      save(analyzed_predictions, word_path)
