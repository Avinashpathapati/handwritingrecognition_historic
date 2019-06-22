import cv2 as cv
import os
import random
import numpy as np
import pandas as pd
import glob

"""Loads greyscale images and returns a list"""


def load_data(path):
    print("loading images...")
    data = []
    names = []

    # Load the grayscale images into the data list.
    path = path + "/"
    for file in os.listdir(path):
        if (str(file).startswith('.') or str(file).endswith('.csv')):
            continue
        image = cv.imread(path + str(file), cv.IMREAD_GRAYSCALE)
        data.append(image)
        names.append(str(file))

    return data, names

def read_image(path):
  return cv.imread(path, cv.IMREAD_UNCHANGED)

def load_images(path):
  print("loading images...")

  # Load the images and their filenames.
  images = []
  filenames = []
  for filename in os.listdir(path + "/"):
    if (str(filename).startswith('.') or str(filename).endswith('.csv')):
        continue
    image = read_image(path + "/" + str(filename))
    images.append(image)
    filenames.append(str(filename))

  return images, filenames


"""Plots an image"""
def plot_opencv(image):
    winname = 'PLOT'
    cv.imshow(winname, image)
    cv.waitKey(0)
    cv.destroyWindow(winname)


def random_color():
    nums = [i for i in range(256)]
    color = []
    random.shuffle(nums)
    color.append(nums[0])

    random.shuffle(nums)
    color.append(nums[0])

    random.shuffle(nums)
    color.append(nums[0])

    return tuple(color)


def invert_image(im):
    im = abs(255 - im)
    im = im / 255

    return im

def make_predictions(model, images, generator=None):
    print("making predictions...")

    if generator is None:
        predictions = model.predict(images)
    else:
        # Make the predictions by performing test time data augmentation using the given generator. For
        # this to make sense the model should have been trained with the same generator.
        predictions = []
        for _ in range(20):
            current_predictions = model.predict_generator(generator.flow(images, batch_size=1, shuffle=False),
                                                          steps=images.shape[0])
            predictions.append(current_predictions)
        predictions = np.mean(predictions, axis=0)

    return predictions


def analyze(predictions, filenames):
    print("analyzing predictions...")

    # Determine the labels and probabilities and store them into a dataframe with
    # the image filenames.
    analyzed_predictions = pd.DataFrame()
    labels = ["Alef", "Ayin", "Bet", "Dalet", "Gimel", "He", "Het", "Kaf", "Kaf-final",
              "Lamed", "Mem", "Mem-medial", "Nun-final", "Nun-medial", "Pe", "Pe-final",
              "Qof", "Resh", "Samekh", "Shin", "Taw", "Tet", "Tsadi-final", "Tsadi-medial",
              "Waw", "Yod", "Zayin"]
    indices = np.argmax(predictions, axis=1)
    probabilities = np.max(predictions, axis=1)
    analyzed_predictions["names"] = filenames
    analyzed_predictions["labels"] = [labels[x] for x in indices]
    analyzed_predictions["probabilities"] = probabilities

    return analyzed_predictions


def save(analyzed_predictions, path):
    # Save the analyzed predictions.
    if not os.path.isdir(str(path)):
        os.mkdir(str(path))

    print("saving analyzed predictions to " + str(path))

    analyzed_predictions.to_csv(str(path) + "/analyzed-predictions.csv", index=False)


def listdir_nohidden(path):
    return glob.glob(os.path.join(path, '*'))