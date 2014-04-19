from src.data import *
print("loading...")
loadData = importDetailed
train = loadData('conll2000/train.txt')
test = loadData('conll2000/test.txt')
test2 = loadData('conll2000/test.txt')
test3 = loadData('conll2000/test.txt')
combined = loadData('conll2000/combined.txt')

allPos = set(word.pos for sentence in train + test for word in sentence)
allChunks = set(word.chunk for sentence in train + test for word in sentence)

testSlices = []
trainSlices = []
numberOfSlices = 10
sliceData(combined, numberOfSlices, trainSlices, testSlices)


from src.hmm import *
from src.prlg import *
from src.pllg import *

print("training...")
predictor = Markov(allPos, allChunks).train(train)
predictor2 = PRLG(allPos, allChunks).train(train)
predictor3 = PLLG(allPos, allChunks).train(train)


# i = 0
# it = predictor.predict(test[i])
# it2 = predictor3.predict(test3[i])
# assert len(it) == len(it2)
# for i in range(0, len(it)):
# 	print(it[i].predicted, it2[i].predicted)
# exit()

print("predicting...")
predictions = []
for sentence in test:
	predictions.append(predictor.predict(sentence))
predictions2 = []
for sentence in test2:
	predictions2.append(predictor2.predict(sentence))
predictions3 = []
for sentence in test3:
	predictions3.append(predictor3.predict(sentence))


# count = 0
# for i in range(0, len(predictions)):
# 	if predictions[i].chunk != predictions2[i].chunk:
# 		count += 1
# print(len(predictions), count)

print("calculating weighted f1...")
print(f1_weighted(predictions, allChunks))
print(f1_weighted(predictions2, allChunks))
print(f1_weighted(predictions3, allChunks))


"""

"""





