from recognizer.extractor import ExtractorByOpening
from recognizer.preprocess import preprocess_single

if __name__ == '__main__':
	ext = ExtractorByOpening(20)
	ext.testing_start()
	# preprocess_single()