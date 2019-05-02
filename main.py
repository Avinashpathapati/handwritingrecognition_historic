from recognizer.extractor import ExtractorByOpening
from recognizer.preprocess import preprocess_single
from recognizer.utility import load_data, save_opencv, plot_opencv, plot_matplotlib


if __name__ == '__main__':
	#extractor = ExtractorByOpening(20)
	#extractor.testing_start()
	
	#"""
	# [Temporary] Saving the images to inspect.
	extractor = ExtractorByOpening(20)
	data = load_data('/home/anpenta/Desktop/handwriting-recognizer/data/image-data/')
	import os
	if not os.path.exists("../output"):
  		os.makedirs("../output")
	i = 0
	for image in data:
		image = extractor.extract_text(image)
		save_opencv(image, '../output/', str(i) + '.jpg')
		i += 1
	#"""
