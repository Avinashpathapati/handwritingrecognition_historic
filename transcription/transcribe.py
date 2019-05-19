from utility import create_image, load_predictions, transcribe

predictions = load_predictions()
word = [x for x in predictions["labels"]]

sentence = []
sentence.append(word)
sentence.append(word)

text = []
text.append(sentence)
text.append(sentence)

transcribe(text)

#create_image('Alef', (50,50)).save("t.png")
#print(u'\u05DE'.encode('utf-8').decode('utf-8'))