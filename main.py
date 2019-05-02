from recognizer.extractor import ExtractorByOpening
from recognizer.preprocess import preprocess_single, preprocess
from recognizer.utility import load_data, save_opencv, plot_opencv, plot_matplotlib, plot_histogram


if __name__ == '__main__':
	#extractor = ExtractorByOpening(20)
	#extractor.testing_start()
	
	#"""
	# [Temporary] Saving the images to inspect.
	extractor = ExtractorByOpening(20)
	data = load_data('/home/anpenta/Desktop/handwriting-recognizer/data/image-data')
	data = [extractor.extract_text(x) for x in data]
	data = preprocess(data)
	
	import os
	if not os.path.exists("../output"):
  		os.makedirs("../output")
	i = 0
	for image in data:
		#plot_histogram(image)
		save_opencv(image, '../output/', str(i) + '.jpg')
		i += 1
	#"""
