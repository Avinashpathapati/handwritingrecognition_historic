from utility import get_font_and_char_map, create_image, load_predictions

predictions = load_predictions()
print(predictions)

exit(1)

font,  char_map = get_font_and_char_map()

#Create a 50x50 image of the Alef token and save it to disk
#To get the raw data cast it to a numpy array
img = create_image('Kaf-final', (50, 50), font, char_map)
img.save('test.png')
