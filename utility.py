import cv2 as cv
import os
import random

"""Loads greyscale images and returns a list"""
def load_data(path):
  print("loading images...")
  data = []
  names = []

  # Load the grayscale images into the data list.
  path = path + "/"
  for file in os.listdir(path):
      if (str(file).startswith('.')):
          continue
      image = cv.imread(path + str(file), cv.IMREAD_GRAYSCALE)
      data.append(image)
      names.append(str(file))

  return data,names


"""Plots an image"""
def plot_opencv(image):
    winname='PLOT'
    cv.imshow(winname,image)
    cv.waitKey(0)
    cv.destroyWindow(winname)


def random_color():
    nums = [ i for i in range(256)]
    color=[]
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