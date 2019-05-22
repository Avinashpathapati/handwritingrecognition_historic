from recognizer.extractor import ExtractorByOpening
from recognizer.preprocess import preprocess_single, preprocess
from recognizer.utility import *
	#load_data, save_opencv, plot_opencv, plot_matplotlib, plot_histogram,invert_image,scatter_plot_of_image
from segmentation.line_segmentation import LineSementation
import cv2 as cv
from matplotlib import pyplot as plt
from recognizer.dominant_colors import DominantColors

from utility.utility import timeit
from segmentation.character_segmentation import CharacterSegmentationMeanShift


if __name__ == '__main__':
	extractor = ExtractorByOpening(20)
	img_and_mask=extractor.testing_start()

	img=preprocess(data=[img_and_mask])

	#img= get_area_filtered_image(img[0])

	# plot_opencv(img[0])
	# save_opencv(img[0],'../data/test/','22.png')
	#save_opencv(img_and_mask[1],'../data/test/','mask_0_test.png')

	#plot_histogram(img_and_mask[0],ylim=[0,7000])

	# dc = DominantColors()
	# img = dc.find_dominant_colors(img_and_mask[0])

	#img=preprocess(data=[img_and_mask])

	#save_opencv(img[0],'../data/test/','binarised_0_test.png')

	#plot_opencv(img[0])




	#plot_matplotlib(img[0])
	#plot_opencv(img[0])
	#plot_opencv(img_and_mask[1])

	line_segmentation = LineSementation()
	line_segmentation.test_segmentation(img[0])

	# character_segmentation = CharacterSegmentationMeanShift(img[0])
	# character_segmentation.plot_characters()

	#visualise_connected_components(img[0])



	#preprocess_single()

	#"""
	# [Temporary] Saving the images to inspect.
	# extractor = ExtractorByOpening(20)
	# data = load_data('/home/anpenta/Desktop/handwriting-recognizer/data/image-data')
	# data = [extractor.extract_text(x) for x in data]
	# data = preprocess(data)
	#
	# import os
	# if not os.path.exists("../output"):
	# 	os.makedirs("../output")
	# i = 0
	# for image in data:
	# 	#plot_histogram(image)
	# 	save_opencv(image, '../output/', str(i) + '.jpg')
	# 	i += 1
	#"""
