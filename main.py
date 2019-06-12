import argparse
from utility import *
from extractor.extractor import ExtractorByOpening
from preprocess.preprocess import *
from segmentation.final_segmentation import *
from segmentation.line_segmentation import LineSegmentation

parameters = {
    'extractor_kernel': [20],
    'preprocess_kernel': [3],
    'whiten_background_kernel': [10],
    'binarization_threshold': [25]
}


def intro():
    print("""
    .=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-.
    |                     ______                     |
    |                  .-"      "-.                  |
    |                 /            \                 |
    |     _          |              |          _     |
    |    ( \         |,  .-.  .-.  ,|         / )    |
    |     > "=._     | )(__/  \__)( |     _.=" <     |
    |    (_/"=._"=._ |/     /\     \| _.="_.="\_)    |
    |           "=._"(_     ^^     _)"_.="           |
    |               "=\__|IIIIII|__/="               |
    |              _.="| \IIIIII/ |"=._              |
    |    _     _.="_.="\          /"=._"=._     _    |
    |   ( \_.="_.="     `--------`     "=._"=._/ )   |
    |    > _.="                            "=._ <    |
    |   (_/       PIRATES OF THE DEAD SEA      \_)   |
    |                                                |
    '-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-='""")


def main():
    intro()

    parser = argparse.ArgumentParser(description='This program performs automatic handwriting recognition on images of the Dead Sea scrolls.')
    parser.add_argument('image_path', help='The path of the folder containing the images to be used.')
    args = parser.parse_args()

    data, names = load_data(args.image_path)
    extractor = ExtractorByOpening(20)
    data = [extractor.extract_text(x) for x in data]

    data = preprocess(data)

    i = 0
    for img in data:
        line_segmentation = LineSegmentation()
        img, line_images = line_segmentation.segment_lines(img)
        name = names[i]
        i += 1
        over_seg_and_graph(line_images, name)


if __name__=='__main__':
    main()