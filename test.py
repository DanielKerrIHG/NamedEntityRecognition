from src.data import *
print("loading...")
loadData = importDetailed
train = loadData('conll2000/train.txt')
test = loadData('conll2000/test.txt')
test2 = loadData('conll2000/test.txt')
combined = loadData('conll2000/combined.txt')

allPos = set(word.pos for sentence in train + test for word in sentence)
allChunks = set(word.chunk for sentence in train + test for word in sentence)

testSlices = []
trainSlices = []
numberOfSlices = 10
sliceData(combined, numberOfSlices, trainSlices, testSlices)


from src.hmm import *
from src.pllg import *

print("training...")
predictor = Markov(allPos, allChunks).train(train)
predictor2 = PLLG(allPos, allChunks).train(train)


# i = 0
# it = predictor.predict(test[i])
# it2 = predictor2.predict(test2[i])
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


# count = 0
# for i in range(0, len(predictions)):
# 	if predictions[i].chunk != predictions2[i].chunk:
# 		count += 1
# print(len(predictions), count)

print("calculating weighted f1...")
print(f1_weighted(predictions, allChunks))
print(f1_weighted(predictions2, allChunks))

print("calculating micro average f1...")
print(f1_micro(predictions, allChunks))
print(f1_micro(predictions2, allChunks))

print("calculating macro average f1...")
print(f1_macro(predictions, allChunks))
print(f1_macro(predictions2, allChunks))

"""
print('========================================')
print("calculating data slices...")
weightedAverage = 0
microAverage = 0
macroAverage = 0
for i in range(numberOfSlices):
	print("calculating using slice " + str(i) + " as testing data:")
	predictor = Markov(allPos, allChunks).train(trainSlices[i])
	predictions = []
	for sentence in testSlices[i]:
		predictions.append(predictor.predict(sentence))
	print("calculating weighted f1...")

	it = f1_weighted(predictions, allChunks)
	print(it)
	weightedAverage += it
	print("calculating micro average f1...")

	alsoIt = f1_micro(predictions, allChunks)
	print(alsoIt)
	microAverage += alsoIt
	print("calculating macro average f1...")
	lastIt = f1_macro(predictions, allChunks)
	print(lastIt)
	macroAverage += lastIt

weightedAverage = weightedAverage / numberOfSlices
microAverage = microAverage / numberOfSlices
macroAverage = macroAverage / numberOfSlices

print("Weighted average: " + str(weightedAverage))
print("micro average: " + str(microAverage))
print("macro average: " + str(macroAverage))
"""





