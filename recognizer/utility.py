# Utility module
# Module to implement utility functions for recognizer.

import matplotlib.pyplot as plt
import argparse
import os
import cv2 as cv
import numpy as np
from PIL import Image
import random

from matplotlib import pyplot as plt


def plot_opencv(image):
  winname='PLOT'
  cv.imshow(winname,image)
  cv.waitKey(0)
  cv.destroyWindow(winname)

def plot_matplotlib(image):
  plt.imshow(Image.fromarray(image))
  plt.show()

def plot_histogram(image,ylim=None,xlim=None):
  plt.hist(image.ravel(), bins=256, range=[0,256])
  plt.xlabel("Pixel value")
  plt.ylabel("Number of pixels")
  if ylim:
    plt.ylim(ylim)

  if xlim:
    plt.xlim(xlim)
  plt.show()



def scatter_plot_of_image(img):
  all_pixels = img.flatten()
  plt.scatter(all_pixels)
  plt.show()




def save_opencv(image, path, name):
  cv.imwrite(os.path.join(path, name), image)

def get_input_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument("-d", "--data", required=True,
                      help="path to input image data directory")
  arguments = vars(parser.parse_args())

  return arguments

def load_data(path):
  print("loading images...")
  data = []
  names = []

  # Load the grayscale images into the data list.
  path = path + "/"
  for file in os.listdir(path):
    if "fused" in str(file):
      image = cv.imread(path + str(file), cv.IMREAD_GRAYSCALE)
      data.append(image)
      names.append(str(file))

  return data,names

def load_single_image(image_path,image_name,load_greyscale=False):
  img = image_path + str(image_name)
  if not os.path.exists(img):
    raise Exception('Path does not exist')

  if load_greyscale:
    image = cv.imread(image_path + str(image_name),cv.IMREAD_GRAYSCALE)
  else:
    #print('in here')
    image = cv.imread(image_path + str(image_name),cv.IMREAD_UNCHANGED)

  #image = cv.imread(image_path + str(image_name), cv.IMREAD_GRAYSCALE)
  return image



def invert_image(im):
    im = abs(255 - im)
    im = im / 255

    return im

def get_area_filtered_image(img):
    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(abs(255 - img), connectivity=8)
    new_output,new_img = filter_connected_components_area_wise(nb_components, output, stats)

    return abs(255-new_img)




def visualise_connected_components(img):
  nb_components, output, stats, centroids = cv.connectedComponentsWithStats(abs(255- img), connectivity=8)

  #print(np.unique(output))
  plot_cc_with_random_color(nb_components,output,save=True,name='1.png')

  new_output,new_img = filter_connected_components_area_wise(nb_components, output,stats)  # will return connected components within height range
  plot_cc_with_random_color(nb_components, new_output,save=True,name='4.png')

  # new_output,new_img = filter_connected_components_height_wise(nb_components,output,stats) #will return connected components within height range
  # plot_cc_with_random_color(nb_components, new_output,save=True,name='2.png')
  # new_output,new_img = filter_connected_components_width_wise(nb_components,new_output,stats)
  # plot_cc_with_random_color(nb_components, new_output,save=True,name='3.png')

  return


def filter_connected_components_height_wise(nb_components,output,stats):
    heights = stats[:, 3]  # Heights of component
    #widths = stats[:, 2]  # Widths of components

    AH = sum(heights) / nb_components  # nb_components is number of components

    subset = np.zeros(output.shape)
    subset_img = np.zeros(output.shape)
    j=0
    for i in range(nb_components):
        H = heights[i]
        if ((0.5 * AH) <= H <= ( 1.5 * AH)):  # condition defined in the paper
            subset[output == i] = j
            subset_img[output == i] = 255
            j = j + 1
    return subset,subset_img

def filter_connected_components_area_wise(nb_components,output,stats):
    #heights = stats[:, 3]  # Heights of component
    areas = stats[:, -1]  # Widths of components

    AA = sum(areas) / nb_components  # nb_components is number of components


    subset = np.zeros(output.shape)
    subset_img = np.zeros(output.shape)
    j=0
    for i in range(nb_components):
        A = areas[i]
        if ((0.05 * AA) <= A <= (0.9 * AA)):  # condition defined in the paper
            subset[output == i] = j
            subset_img[output == i] = 255
            j=j+1
    return subset,subset_img


def filter_connected_components_width_wise(nb_components,output,stats):
    #heights = stats[:, 3]  # Heights of component
    widths = stats[:, 2]  # Widths of components

    AW = sum(widths) / nb_components  # nb_components is number of components


    subset = np.zeros(output.shape)
    subset_img = np.zeros(output.shape)
    j=0
    for i in range(nb_components):
        W = widths[i]
        if ((0.5 * AW) <= W <= (1.5 * AW)):  # condition defined in the paper
            subset[output == i] = j
            subset_img[output == i] = 255
            j=j+1
    return subset,subset_img

def plot_cc_with_random_color(nb_components,output,save=False,name='1.png'):
    preview = np.zeros((output.shape[0], output.shape[1], 3), dtype=np.uint8)

    for i in range(1, nb_components):
        preview[output == i] = random_color()

    plot_opencv(preview)

    if save:
        save_opencv(preview,'../data/test/',name)




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


def draw_image_histogram(image, channels, color='k'):
    hist = cv.calcHist([image], channels, None, [256], [1, 256])
    plt.plot(hist, color=color)
    plt.xlim([0, 256])

def show_color_histogram(image):
    #for i, col in enumerate(['b', 'g', 'r']):
    draw_image_histogram(image, [0], color='b')
    plt.show()
