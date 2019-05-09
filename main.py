from recognizer.extractor import ExtractorByOpening
from recognizer.preprocess import preprocess_single, preprocess
from recognizer.utility import *
from segmentation.line_segmentation import LineSementation


if __name__ == '__main__':
	extractor = ExtractorByOpening(20)
	img_and_mask=extractor.testing_start()
	#plot_opencv(img_and_mask[0])

	#extractor = ExtractorByOpening(50)
	#img_and_mask=extractor.extract_text(img_and_mask[0])

	img=preprocess(data=[img_and_mask])
	img_resized = image_resize(img[0], height=800)
	#mg=preprocess(data=[img_and_mask])



	#plot_matplotlib(img[0])
	plot_opencv(img_resized)
	#plot_opencv(img_and_mask[1])

	# line_segmentation = LineSementation()
	# line_segmentation.test_segmentation(img[0])

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
