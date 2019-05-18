from utility import get_font_and_char_map, create_image, load_predictions, transcribe

predictions = load_predictions()
word = [x for x in predictions["labels"]]
sentence = []
sentence.append(word)
sentence.append(word)

font,  char_map = get_font_and_char_map()

transcribe(sentence, font, char_map)
