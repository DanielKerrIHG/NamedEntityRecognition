from src.data import *
train = importDetailed('conll2000/train.txt')
test = importDetailed('conll2000/test.txt')
combined = importDetailed('conll2000/combined.txt')
detailedPOS = set(word.pos for sentence in train + test for word in sentence)
detailedChunks = set(word.chunk for sentence in train + test for word in sentence)

testSlices = []
trainSlices = []
numberOfSlices = 10
sliceData(combined, numberOfSlices, trainSlices, testSlices)


print("loaded")
from src.hmm import *
from src.prlg import *


predictor = Markov(detailedPOS, detailedChunks).train(train)
predictions = []
for sentence in test:
	predictions.append(predictor.viterbi(sentence))

predictor2 = PRLG(detailedPOS, detailedChunks).train(train)



print("calculating weighted f1...")

it = f1_weighted(predictions, detailedChunks)
print(it)

print("calculating micro average f1...")

alsoIt = f1_micro(predictions, detailedChunks)
print(alsoIt)

print("calculating macro average f1...")
lastIt = f1_macro(predictions, detailedChunks)
print(lastIt)

"""
print('========================================')
print("calculating data slices...")
weightedAverage = 0
microAverage = 0
macroAverage = 0
for i in range(numberOfSlices):
	print("calculating using slice " + str(i) + " as testing data:")
	predictor = Markov(detailedPOS, detailedChunks).train(trainSlices[i])
	predictions = []
	for sentence in testSlices[i]:
		predictions.append(predictor.viterbi(sentence))
	print("calculating weighted f1...")

	it = f1_weighted(predictions, detailedChunks)
	print(it)
	weightedAverage += it
	print("calculating micro average f1...")

	alsoIt = f1_micro(predictions, detailedChunks)
	print(alsoIt)
	microAverage += alsoIt
	print("calculating macro average f1...")
	lastIt = f1_macro(predictions, detailedChunks)
	print(lastIt)
	macroAverage += lastIt

weightedAverage = weightedAverage / numberOfSlices
microAverage = microAverage / numberOfSlices
macroAverage = macroAverage / numberOfSlices

print("Weighted average: " + str(weightedAverage))
print("micro average: " + str(microAverage))
print("macro average: " + str(macroAverage))
"""





