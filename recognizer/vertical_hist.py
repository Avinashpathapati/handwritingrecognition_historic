#!/usr/bin/env python3

import cv2
import numpy as np
import math
import skfuzzy as fuzz
from numpy import array
import matplotlib.pyplot as plt

from utility import load_data


def show_proj(proj, shape):

  # Create output image same height as text, 500 px wide
  m = np.max(proj)
  w = 500
  result = np.zeros((proj.shape[0],500))

  # Draw a line for each col
  for col in range(shape[1]):
    cv2.line(result, (col,0), (col, int(proj[col]*w/m)), (255,255,255), 1)
  # Save result
  cv2.imwrite('result.png', result)

def gap_stats_and_word_lengths(image):

  #computing the gap statistics
  # Load as greyscale
  word_len_lst = []
  gap_len_lst = []

  for im in image:
    # Invert
    im = 255 - im

    # Calculate vertical projection
    proj = np.sum(im,0)
    word_st = -1
    word_end = -1
    gp_st = -1
    gp_st = -1
    gp_end = -1

    #computing the word and gap lengths
    for col in range(proj.shape[0]):
      if proj[col] == 0:
        word_len = math.sqrt(((word_end-word_st+1))**2)
        if not word_st == -1:
          word_len_lst.append(word_len)
        word_st = -1
        if gp_st == -1:
          gp_st = col
        gp_end = col

      else:
        gap_len = math.sqrt(((gp_end-gp_st+1))**2)
        if not gp_st == -1:
          gap_len_lst.append(gap_len)
        gp_st = -1
        if word_st == -1:
          word_st = col

        word_end = col


    if word_st != -1 and not proj[proj.shape[0]-1] ==0:
      word_len_lst.append(math.sqrt((word_end-word_st+1)**2))

    if gp_st != -1 and proj[proj.shape[0]-1] ==0:
      gap_len_lst.append(math.sqrt((gp_end-gp_st)**2))



  word_len_np = array(word_len_lst)
  gap_len_np = array(gap_len_lst)

  word_len_np = np.reshape(word_len_np, (1, word_len_np.shape[0]))
  gap_len_np = np.reshape(gap_len_np, (1,gap_len_np.shape[0]))
  
  return word_len_np,gap_len_np

  
def word_seg(image, w_cntr, g_cntr, w_u_orig, g_u_orig):

  # Load as greyscale
  #clustering a space as word segment or character segment
  ct = 0;
  for im in image:
    ct = ct+1;
    # Invert
    im = 255 - im

    # Calculate vertical projection
    proj = np.sum(im,0)

    word_st = -1
    word_end = -1
    gp_st = -1
    gp_st = -1
    gp_end = -1

    #computing the word and gap lengths
    for col in range(proj.shape[0]):
      if proj[col] == 0:
        word_len = ((word_end-word_st+1))**2
        if not word_st == -1:
          #word_len_lst.append(word_len)
          print('hell')
        word_st = -1
        if gp_st == -1:
          gp_st = col
        gp_end = col

      else:
        gap_len = ((gp_end-gp_st+1))**2
        if not gp_st == -1:
          gap_len_arr = np.reshape(np.array([gap_len]),(1,1))
          u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(gap_len_arr, g_cntr, 2, error=0.00001, maxiter=1000)
          label = np.argmax(u, axis = 0)
          big_gap_clust_label = np.argmax(g_cntr, axis=0)
          if label == big_gap_clust_label:
            cv2.line(im, (gp_st,0), (gp_st, im.shape[0]), (255,0,0), 1)
            cv2.line(im, (gp_end,0), (gp_end, im.shape[0]), (255,0,0), 1)
          #gap_len_lst.append(gap_len)
        
        gp_st = -1
        if word_st == -1:
          word_st = col

        word_end = col


    # if word_st != -1 and not proj[proj.shape[0]-1] ==0:
    #   print('hell')
    #   word_len_lst.append((word_end-word_st+1)**2)

    if gp_st != -1 and proj[proj.shape[0]-1] ==0:
      cv2.line(im, (gp_st,0), (gp_st, im.shape[0]), (255,0,0), 1)

    # Save result
    cv2.imwrite('word_seg_'+str(ct)+'.png', im)




image_data = load_data('/Users/sandy/Downloads/heb-crop', 'crop')
print(image_data)
if not len(image_data) == 0:
  word_len_np, gap_len_np = gap_stats_and_word_lengths(image_data)
  w_cntr, w_u_orig, _, _, _, _, _ = fuzz.cluster.cmeans(word_len_np, 2, 2, error=0.00001, maxiter=1000)
  g_cntr, g_u_orig, _, _, _, _, _ = fuzz.cluster.cmeans(gap_len_np, 2, 2, error=0.00001, maxiter=1000)
  word_seg(image_data, w_cntr, g_cntr, w_u_orig, g_u_orig)
else:
  print('no images found')




















