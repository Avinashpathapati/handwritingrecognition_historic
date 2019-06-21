# Classify module
# Author: Andreas Pentaliotis
# Module to implement character classification.

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import os

from utility import load_images, make_predictions, analyze, save, listdir_nohidden
from classification.preprocessing import preprocess


def classify(scrolls_path):
    cnn = load_model("./classification/models/cnn.h5")
    generator = ImageDataGenerator(zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1, rotation_range=5)
    for scroll_folder in listdir_nohidden(scrolls_path):
        for line_directory in listdir_nohidden(scroll_folder):
            for word_directory in listdir_nohidden(line_directory):
                # Load and preprocess the images.
                images, filenames = load_images(word_directory)
                images = preprocess(images)

                predictions = make_predictions(cnn, images, generator=generator)

                analyzed_predictions = analyze(predictions, filenames)
                word_path = word_directory.split(".png")[0]
                save(analyzed_predictions, word_path)
