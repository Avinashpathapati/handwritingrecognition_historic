#!/usr/bin/env python3

import cv2 as cv
import numpy as np
import math
import sys
import os
#from utility import load_data


def clean_img(image):
    kernel_er_size = 3
    kernel_er = np.ones((kernel_er_size, kernel_er_size), np.uint8)
    image = cv.erode(image, kernel_er, iterations=2)

    kernel_dil_size = 2
    kernel_dil = np.ones((kernel_dil_size, kernel_dil_size), np.uint8)
    dilation = cv.dilate(image, kernel_dil, iterations=5)
    # find all your connected components (white blobs in your image)
    # print(image.shape)
    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(dilation, connectivity=8)
    sizes = stats[1:, -1];
    nb_components = nb_components - 1

    # # minimum size of particles we want to keep (number of pixels)
    # # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    # min_size = 150

    # # your answer image
    img2 = np.zeros((output.shape))
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255

    # #opening = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)
    # plt.imshow(Image.fromarray(img2))
    # plt.show()
    return img2


def find_y_loc_seg(h_proj):
    min_h = 0
    for i, vp_val in enumerate(h_proj):
        if vp_val == 0:
            min_h = i
        else:
            break

    max_h = len(h_proj) - 1
    for i, vp_val in reversed(list(enumerate(h_proj))):
        if vp_val == 0:
            max_h = i
        else:
            break

    return min_h, max_h


def isValid(x, y, st_y, end_y, end_x):
    if x < 0 or x > end_x:
        return False

    if y < st_y or y > end_y:
        return False

    return True


def proj_sum_zero(proj_new, st, end):
    for i in range(st, end + 1):
        if proj_new[i] > 0:
            return False
    return True


def find_left_right_start(proj,st,end):
    left = 0
    right = proj.shape[0] -1

    for i in range(st,end):
        if not proj[i] == 0:
            left = i
            break

    for i in reversed(list(range(st,end))):
        if not proj[i] == 0:
            right = i
            break


    return left,right

def find_min_cur_level(im, cur_depth, st, end):
    min_val = float('inf')
    min_x = -1
    for i in range(st, end + 1):
        if im[cur_depth][i] < min_val:
            min_val = im[cur_depth][i]
            min_x = 0

    return min_x


def gap_stats_and_word_lengths(proj):
    # computing the gap statistics
    # Load as greyscale
    word_len_lst = []
    word_loc_list = []
    gap_len_lst = []
    gap_loc_list = []
    word_st = -1
    word_end = -1
    gp_st = -1
    gp_st = -1
    gp_end = -1

    # computing the word and gap lengths
    for col in range(proj.shape[0]):
        if proj[col] == 0:
            word_len = math.sqrt(((word_end - word_st + 1)) ** 2)
            if not word_st == -1:
                word_len_lst.append(word_len)
                word_loc_list.append(word_st)
                word_loc_list.append(word_end)
            word_st = -1
            if gp_st == -1:
                gp_st = col
            gp_end = col

        else:
            gap_len = abs(gp_end - gp_st + 1)
            if not gp_st == -1 and gap_len > 10:
                if not len(word_len_lst) == 0:
                    gap_len_lst.append(gap_len)
                    gap_loc_list.append(gp_st)
            gp_st = -1
            if word_st == -1:
                word_st = col

            word_end = col

    if word_st != -1 and not proj[proj.shape[0] - 1] == 0:
        word_len_lst.append(math.sqrt((word_end - word_st + 1) ** 2))
        word_loc_list.append(word_st)
        word_loc_list.append(word_end)

    if gp_st != -1 and proj[proj.shape[0] - 1] == 0:
        gap_len_lst.append(math.sqrt((gp_end - gp_st) ** 2))

    return gap_loc_list, word_loc_list


def dp_find_final_path(cost_path_arr, short_path_arr, cur_in, min_in):
    for i in reversed(list(range(cur_in+1))):
        if i >= 1:
            min_new_in = min_in
            min_val = cost_path_arr[i - 1][min_in]
            if min_in - 1 >= 0:
                if cost_path_arr[i - 1][min_in - 1] < min_val:
                    min_new_in = min_in - 1
                    min_val = cost_path_arr[i - 1][min_in - 1]
            if min_in + 1 < cost_path_arr.shape[1]:
                if cost_path_arr[i - 1][min_in + 1] < min_val:
                    min_new_in = min_in + 1
                    min_val = cost_path_arr[i - 1][min_in + 1]
            short_path_arr[i - 1] = min_new_in
            min_in = min_new_in


def find_recursive_path(cost_path_arr, short_path_arr, cur_in, min_in):
    min_val = float('inf')
    min_new_in = -1
    if cur_in >= 1:
        min_new_in = min_in
        min_val = cost_path_arr[cur_in - 1][min_in]
        if min_in - 1 >= 0:
            if cost_path_arr[cur_in - 1][min_in - 1] < min_val:
                min_new_in = min_in - 1
                min_val = cost_path_arr[cur_in - 1][min_in - 1]
        if min_in + 1 < cost_path_arr.shape[1]:
            if cost_path_arr[cur_in - 1][min_in + 1] < min_val:
                min_new_in = min_in + 1
                min_val = cost_path_arr[cur_in - 1][min_in + 1]
        short_path_arr[cur_in - 1] = min_new_in
        find_recursive_path(cost_path_arr, short_path_arr, cur_in - 1, min_new_in)



def find_path_shortest(cost_path_arr, short_path_arr):
    min_cost = float('inf')
    min_index = -1

    for i in range(0, cost_path_arr.shape[1]):
        if cost_path_arr[cost_path_arr.shape[0] - 1][i] < min_cost:
            min_cost = cost_path_arr[cost_path_arr.shape[0] - 1][i]
            min_index = i

    short_path_arr[cost_path_arr.shape[0] - 1] = min_index
    #find_recursive_path(cost_path_arr, short_path_arr, cost_path_arr.shape[0] - 1, min_index)
    dp_find_final_path(cost_path_arr, short_path_arr, cost_path_arr.shape[0] - 1, min_index)



def path_search_dp(im, st, end):
    short_path_arr = [-1] * im.shape[0]

    cost_path_arr = np.full((im.shape[0], (end - st) + 1), float('inf'))

    for x in range(0, cost_path_arr.shape[0]):
        # for x in range(0,(y_max-y_min)+1):
        for y in range(0, (end - st) + 1):
            if x == 0:
                cost_path_arr[x][y] = im[x][st + y]
            else:
                cost_path_arr[x][y] = cost_path_arr[x - 1][y]
                if (y - 1) >= 0:
                    cost_path_arr[x][y] = min(cost_path_arr[x][y], cost_path_arr[x - 1][y - 1])
                if (y + 1) < cost_path_arr.shape[1]:
                    cost_path_arr[x][y] = min(cost_path_arr[x][y], cost_path_arr[x - 1][y + 1])

                cost_path_arr[x][y] = cost_path_arr[x][y] + im[x][st + y]

    find_path_shortest(cost_path_arr, short_path_arr)
    #print(short_path_arr)
    return short_path_arr

    # for y in range(0, cost_path_arr.shape[0]):
    #   for x in range(0, cost_path_arr.shape[1]):
    #       if y == 0:
    #         cost_path_arr[y][x] = im[st+]


def rec_path_search(im, v_x, v_y, st, end, prev_cost, path_arr):
    global cost, min_cost_path_arr
    cur_cost = prev_cost + im[v_x][v_y]
    path_arr[v_x] = v_y
    # print('processing'+str(v_x)+" "+str(v_y))
    if v_x == im.shape[0] - 1 and cur_cost < cost:
        # print('the image paths')
        print(path_arr[:])
        cost = cur_cost
        min_cost_path_arr = path_arr[:]

    if isValid(v_x + 1, v_y + 1, st, end, im.shape[0] - 1):
        rec_path_search(im, v_x + 1, v_y + 1, st, end, cur_cost, path_arr)

    if isValid(v_x, v_y + 1, st, end, im.shape[0] - 1):
        rec_path_search(im, v_x, v_y + 1, st, end, cur_cost, path_arr)

    if isValid(v_x - 1, v_y + 1, st, end, im.shape[0] - 1):
        rec_path_search(im, v_x - 1, v_y + 1, st, end, cur_cost, path_arr)


# # Function to perform multi stage graph search
def multi_stage_graph(st, end, im):
    # Mark all the vertices as not visited
    # depth_min_cost = [sys] * im.shape[0]+1
    # depth_min_path = [-1] * im.shape[0]+1
    #global cost, min_cost_path_arr
    min_each_path_cost = float('inf')
    # for v in range(st, end):
    #   #dist, cost =rec_path_search(im,v,0,st, end,depth_min_cost,depth_min_path,im[v][0])
    #   print('started processing '+str(v))
    #   cost = sys.maxint
    #   path_arr = [-1] * im.shape[0]
    short_path_arr = path_search_dp(im, st, end - 1)
    # if cost < min_each_path_cost:
    #   min_each_path_cost = cost
    #   min_final_path = min_cost_path_arr

    # print('---------')
    # print('min cost path arr found')
    # print(min_final_path)
    return short_path_arr


def draw_cv_line(short_path_arr, im, st):
    for i in range(0, len(short_path_arr) - 1):
        cv.line(im, (st + short_path_arr[i], i), (st + short_path_arr[i + 1], i + 1), (255, 0, 0), 1)


def plot_word_segs(im, word_gaps):
    for gap_loc in word_gaps:
        cv.line(im, (gap_loc, 0), (gap_loc, im.shape[0]), (255, 0, 0), 1)


def extract_char_save_fold(short_path_arr, im, st, end_seg, line_num, word_num, x_min, x_max, path_present, scrol_name, x_min_ch_st,x_max_ch_end):
    save_path = os.path.join('tmp',str(scrol_name.split('.')[0]), str(line_num), str(word_num))

    if path_present:        
        end = st + max(short_path_arr)
        st_2_seg = st + min(short_path_arr)
        char_im = np.zeros((x_max, end - st + 1), dtype=int)
        char_im2 = np.zeros((x_max, end_seg - st_2_seg + 1), dtype=int)
        for i in range(0, len(short_path_arr) - 1):
            char_im[i, 0:short_path_arr[i] + 1] = im[i, st:st + short_path_arr[i] + 1]
            char_im2[i, 0:end_seg - (st + short_path_arr[i])] = im[i, st + short_path_arr[i]:end_seg]

        # if not max(short_path_arr) == -1:
        #   char_im = np.zeros((y_max,end-st+1),dtype=int)
        #   print(char_im)
        #   for i in range(0,len(short_path_arr)-1):
        #     char_im[i,0:short_path_arr[i]+1] = im[i,st:st+short_path_arr[i]+1]
        # print('-----------extract')
        # print(st)
        # print(end)
        # print(st_2_seg)
        # print(end_seg-1)
        # print(max(short_path_arr))

        # print('---------')
        if (end - st) >= 15 :
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            
            char_im_copy = np.zeros(((x_max_ch_end-x_min_ch_st)+10,char_im.shape[1]+10))
            char_im_copy[5:-5,5:-5] = char_im[x_min_ch_st:x_max_ch_end]
            cv.imwrite(os.path.join(save_path, 'char_' + 'col_st_'+str(st) + '_col_end_'+str(end) + '_row_st_'+str(x_min) + '_row_end_'+str(x_max) + '.png'),
                   char_im_copy)
        if (end_seg - st_2_seg) >= 15:

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            char_im2_copy = np.zeros(((x_max_ch_end-x_min_ch_st)+10,char_im2.shape[1]+10))
            char_im2_copy[5:-5,5:-5] = char_im2[x_min_ch_st:x_max_ch_end]
            cv.imwrite(
            os.path.join(save_path, 'char_' +'col_st_'+str(st_2_seg) + '_col_end_'+str(end_seg - 1) + '_row_st_'+str(x_min) + '_row_end_'+str(x_max) + '.png'),
            char_im2_copy)
    else:
        # print('-----------extract no path')
        # print(st)
        # print(end_seg)
        # print('---------')
        char_im = im[0:x_max, st:end_seg]
        if (end_seg - st) >= 15:
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            char_im_copy = np.zeros(((x_max_ch_end-x_min_ch_st)+10,char_im.shape[1]+10))
            char_im_copy[5:-5,5:-5] = char_im[x_min_ch_st:x_max_ch_end]
            cv.imwrite(os.path.join(save_path, 'char_' + 'col_st_'+str(st) + '_col_end_'+str(end_seg) + '_row_st_'+str(x_min) + '_row_end_'+str(x_max)+ '.png'),
                   char_im_copy)


def over_seg_and_graph(images, scrol_name):
    # computing the gap statistics
    # Load as greyscale
    im_ct = 0
    gap_th = 40
    print('started processing scrol ', str(scrol_name))
    for im in images:
        char_seg_col = []
        im_ct = im_ct + 1;
        print('started processing line ', str(im_ct))
        # Invert
        im.astype(int)
        im[im==1] = 0
        #im = 255 - im
        #im = clean_img(im)
        # Calculate vertical projection
        proj = np.sum(im, 0)
        #finding left right extreme as computing mean directly on the image returning very less value
        left,right = find_left_right_start(proj,0,proj.shape[0])
        # Calculate horizontal projection
        h_proj = np.sum(im, 1)
        y_min, y_max = find_y_loc_seg(h_proj)
        word_gaps, word_loc = gap_stats_and_word_lengths(proj)

        proj_val = 0.50 * np.mean(proj[left:right+1])
        th_val = proj_val
        gap_min_ind = 0
        for i in range(1, len(word_loc), 2):
            w_st = word_loc[i - 1]
            w_end = word_loc[i]
            cur_seg_gap = w_st
            char_seg_col.append(w_st)
            j=w_st
            while j in range(w_st, w_end):
                if proj[j] < th_val and (j - cur_seg_gap) >= gap_th: #and (w_end - j) >= gap_th:
                    char_seg_col.append(j)
                    index_skip_zeros = j+1
                    for k in range(j+1,w_end):
                        if not proj[k] == 0:
                            index_skip_zeros = k
                            break;
                    char_seg_col.append(index_skip_zeros)
                    cur_seg_gap = index_skip_zeros
                    j = index_skip_zeros
                else:
                    j = j+1
            char_seg_col.append(w_end)

        
        # for i,vp_val in enumerate(proj):
        #   if vp_val < th_val:
        #     if (i - gap_min_ind) >= gap_th:
        #       char_seg_col.append(i)
        #       gap_min_ind = i
        # cv.line(im, (i,0), (i, im.shape[0]), (255,0,0), 1)

        # if word_st != -1 and not proj[proj.shape[0]-1] ==0:
        #   print('hell')
        #   word_len_lst.append((word_end-word_st+1)**2)
        # Save result
        #plot_word_segs(im, word_loc)
        seg_st = -1
        cur_word_num = 0
        char_seg_col_new = []
        for i in range(1, len(char_seg_col), 2):
            zero_flag = proj_sum_zero(proj, char_seg_col[i - 1], char_seg_col[i])
            if not zero_flag:
                # find_lines_using_graph(char_seg_col[i-1], char_seg_col[i], im)
                if cur_word_num < len(word_gaps) and char_seg_col[i - 1] > word_gaps[cur_word_num]:
                    cur_word_num = cur_word_num + 1

                cost = float('inf')
                # cv.line(im, (char_seg_col[i-1],y_min), (char_seg_col[i-1], y_max), (255,0,0), 1)
                # cv.line(im, (char_seg_col[i],y_min), (char_seg_col[i], y_max), (255,0,0), 1)
                if char_seg_col[i] -5 > char_seg_col[i-1]:
                    # print('-----------before extract')
                    # print(char_seg_col[i-1])
                    # print(char_seg_col[i])
                    # print('---------')
                    short_path_arr = multi_stage_graph(char_seg_col[i - 1], char_seg_col[i] -5, im)
                    extract_char_save_fold(short_path_arr, im, char_seg_col[i - 1], char_seg_col[i] + 1, im_ct,
                                           (len(word_gaps) - cur_word_num)+1 , 0, im.shape[0], True, scrol_name,y_min,y_max)
                    #draw_cv_line(short_path_arr,im,char_seg_col[i-1])
                else :
                    short_path_arr = []
                    extract_char_save_fold(short_path_arr, im, char_seg_col[i - 1], char_seg_col[i] + 1, im_ct,
                                           (len(word_gaps) - cur_word_num)+1 , 0, im.shape[0], False, scrol_name,y_min,y_max)

                
                cv.line(im, (char_seg_col[i],0), (char_seg_col[i], im.shape[0]), (255,0,0), 1)

                # short_path_arr = multi_stage_graph(char_seg_col[i-1], char_seg_col[i]+1, im,y_min,y_max)
                #if (char_seg_col[i - 1] + short_path_arr[len(short_path_arr) / 2]) - char_seg_col[i - 1] > 30:
                    # draw_cv_line(short_path_arr,im,char_seg_col[i-1])
                #extract_char_save_fold(short_path_arr, im, char_seg_col[i - 1], char_seg_col[i] + 1, im_ct,
                                           #(len(word_gaps) - cur_word_num)+1 , 0, im.shape[0], True, scrol_name)
                #else:
                    #extract_char_save_fold(short_path_arr, im, char_seg_col[i - 1], char_seg_col[i] + 1, im_ct,
                                           #(len(word_gaps) - cur_word_num)+1 , 0, im.shape[0], False, scrol_name)

        print('finished processing line ', str(im_ct))
        cv.imwrite('word_seg_'+str(scrol_name)+str(im_ct)+'.png', im)

    print('finished processing scrol ', str(scrol_name))
    # multi_stage_graph(0, 3, graph)

    # for i in range(1,len(char_seg_col)-1):
    #   zero_flag = proj_sum_zero(proj,char_seg_col[i-1],char_seg_col[i])
    #   if not zero_flag :
    #     zero_flag = proj_sum_zero(proj,char_seg_col[i],char_seg_col[i+1])
    #   if not zero_flag:
    #     char_seg_col_new.append(i)
    #     #cv.line(im, (char_seg_col[i],0), (char_seg_col[i], im.shape[0]), (255,0,0), 1)

    # seg_end = seg_col
    # if seg_st != -1:
    #   print('----------')
    #   print(seg_st)
    #   print(seg_end)
    #   print('----------')
    # seg_st = seg_col
    


def word_seg(image, w_cntr, g_cntr, w_u_orig, g_u_orig):
    # Load as greyscale
    # clustering a space as word segment or character segment
    ct = 0;
    for im in image:
        ct = ct + 1;
        # Invert
        im = 255 - im
        im = clean_img(im)

        # Calculate vertical projection
        proj = np.sum(im, 0)
        proj_val = np.mean(proj)
        th_val = 0.34 * proj_val
        for i, vp_val in enumerate(proj):
            if vp_val < th_val:
                cv.line(im, (i, 0), (i, im.shape[0]), (255, 0, 0), 1)

        # if word_st != -1 and not proj[proj.shape[0]-1] ==0:
        #   print('hell')
        #   word_len_lst.append((word_end-word_st+1)**2)
        # Save result
        cv.imwrite('word_seg_' + str(ct) + '.png', im)


# image_data = load_data('/Users/sandy/Downloads/heb-crop', 'word')
# short_path_arr = []
# cost = sys.maxint
# if not len(image_data) == 0:
#   over_seg_and_graph(image_data,"a")
# else:
#   print('no images found')




















