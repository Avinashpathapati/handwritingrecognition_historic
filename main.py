from recognizer.extractor import ExtractorByOpening
from recognizer.preprocess import preprocess
from recognizer.utility import *
from segmentation.line_segmentation import LineSementation
import os

def start():
	input_dir= '../data/test/tmpsingle'
	output_dir = '../output4'
	data = load_data(input_dir)
	extractor = ExtractorByOpening(20)
	data = [extractor.extract_text(x) for x in data]
	data = preprocess(data)

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	i = 0

	data_new=[]
	for img in data:
		line_segmentation = LineSementation()
		img,line_images=line_segmentation.segment_lines(img)
		data_new.append(img)
		save_opencv(img,output_dir, str(i) + '.jpg')
		i += 1


if __name__ == '__main__':
	start()
