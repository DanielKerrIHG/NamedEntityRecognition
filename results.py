from src.data import *
print("loading...")
loadData = importDetailed
train = loadData('conll2000/train.txt')
test = loadData('conll2000/test.txt')
combined0 = loadData('conll2000/combined.txt')
combined1 = loadData('conll2000/combined.txt')
combined2 = loadData('conll2000/combined.txt')
combined3 = loadData('conll2000/combined.txt')
combined4 = loadData('conll2000/combined.txt')
combined5 = loadData('conll2000/combined.txt')

allPos = set(word.pos for sentence in train + test for word in sentence)
allChunks = set(word.chunk for sentence in train + test for word in sentence)

print("data processing...")
#hmm vanilla
testSlicesHMM = []
trainSlicesHMM = []
numberOfSlices = 10
sliceData(combined0, numberOfSlices, trainSlicesHMM, testSlicesHMM)

#PRLG vanilla
testSlicesPRLG = []
trainSlicesPRLG = []
numberOfSlices = 10
sliceData(combined1, numberOfSlices, trainSlicesPRLG, testSlicesPRLG)

#PLLG vanilla
testSlicesPLLG = []
trainSlicesPLLG = []
numberOfSlices = 10
sliceData(combined2, numberOfSlices, trainSlicesPLLG, testSlicesPLLG)

#hmm semi-supervised
testSlicesSSHMM = []
trainSlicesSSHMM = []
numberOfSlices = 10
sliceData(combined3, numberOfSlices, trainSlicesSSHMM, testSlicesSSHMM)

#PRLG semi-supervised
testSlicesSSPRLG = []
trainSlicesSSPRLG = []
numberOfSlices = 10
sliceData(combined4, numberOfSlices, trainSlicesSSPRLG, testSlicesSSPRLG)

#PLLG semi-supervised
testSlicesSSPLLG = []
trainSlicesSSPLLG = []
numberOfSlices = 10
sliceData(combined5, numberOfSlices, trainSlicesSSPLLG, testSlicesSSPLLG)

from src.hmm import *
from src.pllg import *
from src.prlg import *
from src.semi import *
from src.semipllg import *
from src.semiprlg import *


#HMM Vanilla
print('================HMM Vanilla========================')
print("calculating data slices...")
weightedAverage = 0
microAverage = 0
macroAverage = 0
for i in range(numberOfSlices):
	print("calculating using slice " + str(i) + " as testing data:")
	predictor = Markov(allPos, allChunks).train(trainSlicesHMM[i])
	predictions = []
	for sentence in testSlicesHMM[i]:
		predictions.append(predictor.predict(sentence))
	print("calculating weighted f1...")

	it = f1_weighted(predictions, allChunks)
	print(it)
	print("-----------------------------")
	weightedAverage += it
	print("calculating micro average f1...")

	alsoIt = f1_micro(predictions, allChunks)
	print(alsoIt)
	print("-----------------------------")
	microAverage += alsoIt
	print("calculating macro average f1...")
	lastIt = f1_macro(predictions, allChunks)
	print(lastIt)
	print("-----------------------------")
	macroAverage += lastIt

weightedAverage = weightedAverage / numberOfSlices
microAverage = microAverage / numberOfSlices
macroAverage = macroAverage / numberOfSlices

print("HMM Weighted average: " + str(weightedAverage))
print("HMM micro average: " + str(microAverage))
print("HMM macro average: " + str(macroAverage))

#PLLG Vanilla
print('================PLLG Vanilla========================')
print("calculating data slices...")
weightedAverage = 0
microAverage = 0
macroAverage = 0
for i in range(numberOfSlices):
	print("calculating using slice " + str(i) + " as testing data:")
	predictor = PLLG(allPos, allChunks).train(trainSlicesPLLG[i])
	predictions = []
	for sentence in testSlicesPLLG[i]:
		predictions.append(predictor.predict(sentence))
	print("calculating weighted f1...")

	it = f1_weighted(predictions, allChunks)
	print(it)
	print("-----------------------------")
	weightedAverage += it
	print("calculating micro average f1...")

	alsoIt = f1_micro(predictions, allChunks)
	print(alsoIt)
	print("-----------------------------")
	microAverage += alsoIt
	print("calculating macro average f1...")
	lastIt = f1_macro(predictions, allChunks)
	print(lastIt)
	print("-----------------------------")
	macroAverage += lastIt

weightedAverage = weightedAverage / numberOfSlices
microAverage = microAverage / numberOfSlices
macroAverage = macroAverage / numberOfSlices

print("PLLG Weighted average: " + str(weightedAverage))
print("PLLG micro average: " + str(microAverage))
print("PLLG macro average: " + str(macroAverage))

#PRLG Vanilla
print('================PRLG Vanilla========================')
print("calculating data slices...")
weightedAverage = 0
microAverage = 0
macroAverage = 0
for i in range(numberOfSlices):
	print("calculating using slice " + str(i) + " as testing data:")
	predictor = PRLG(allPos, allChunks).train(trainSlicesPRLG[i])
	predictions = []
	for sentence in testSlicesPRLG[i]:
		predictions.append(predictor.predict(sentence))
	print("calculating weighted f1...")

	it = f1_weighted(predictions, allChunks)
	print(it)
	print("-----------------------------")
	weightedAverage += it
	print("calculating micro average f1...")

	alsoIt = f1_micro(predictions, allChunks)
	print(alsoIt)
	print("-----------------------------")
	microAverage += alsoIt
	print("calculating macro average f1...")
	lastIt = f1_macro(predictions, allChunks)
	print(lastIt)
	print("-----------------------------")
	macroAverage += lastIt

weightedAverage = weightedAverage / numberOfSlices
microAverage = microAverage / numberOfSlices
macroAverage = macroAverage / numberOfSlices

print("PRLG Weighted average: " + str(weightedAverage))
print("PRLG micro average: " + str(microAverage))
print("PRLG macro average: " + str(macroAverage))

#SSHMM
print('================semi-supervised HMM========================')
print("calculating data slices...")
weightedAverage = 0
microAverage = 0
macroAverage = 0
for i in range(numberOfSlices):
	print("calculating using slice " + str(i) + " as testing data:")
	predictor = semiTrain(trainSlicesSSHMM[i], allPos, allChunks)
	predictions = []
	for sentence in testSlicesSSHMM[i]:
		predictions.append(predictor.predict(sentence))
	print("calculating weighted f1...")

	it = f1_weighted(predictions, allChunks)
	print(it)
	print("-----------------------------")
	weightedAverage += it
	print("calculating micro average f1...")

	alsoIt = f1_micro(predictions, allChunks)
	print(alsoIt)
	print("-----------------------------")
	microAverage += alsoIt
	print("calculating macro average f1...")
	lastIt = f1_macro(predictions, allChunks)
	print(lastIt)
	print("-----------------------------")
	macroAverage += lastIt

weightedAverage = weightedAverage / numberOfSlices
microAverage = microAverage / numberOfSlices
macroAverage = macroAverage / numberOfSlices

print("SSHMM Weighted average: " + str(weightedAverage))
print("SSHMM micro average: " + str(microAverage))
print("SSHMM macro average: " + str(macroAverage))

#SSPLLG
print('================semi-supervised PLLG========================')
print("calculating data slices...")
weightedAverage = 0
microAverage = 0
macroAverage = 0
for i in range(numberOfSlices):
	print("calculating using slice " + str(i) + " as testing data:")
	predictor = semiTrain(trainSlicesSSPLLG[i], allPos, allChunks)
	predictions = []
	for sentence in testSlicesSSPLLG[i]:
		predictions.append(predictor.predict(sentence))
	print("calculating weighted f1...")

	it = f1_weighted(predictions, allChunks)
	print(it)
	print("-----------------------------")
	weightedAverage += it
	print("calculating micro average f1...")

	alsoIt = f1_micro(predictions, allChunks)
	print(alsoIt)
	print("-----------------------------")
	microAverage += alsoIt
	print("calculating macro average f1...")
	lastIt = f1_macro(predictions, allChunks)
	print(lastIt)
	print("-----------------------------")
	macroAverage += lastIt

weightedAverage = weightedAverage / numberOfSlices
microAverage = microAverage / numberOfSlices
macroAverage = macroAverage / numberOfSlices

print("SSPLLG Weighted average: " + str(weightedAverage))
print("SSPLLG micro average: " + str(microAverage))
print("SSPLLG macro average: " + str(macroAverage))

#SSPRLG
print('================semi-supervised PRLG========================')
print("calculating data slices...")
weightedAverage = 0
microAverage = 0
macroAverage = 0
for i in range(numberOfSlices):
	print("calculating using slice " + str(i) + " as testing data:")
	predictor = semiTrain(trainSlicesSSPRLG[i], allPos, allChunks)
	predictions = []
	for sentence in testSlicesSSPRLG[i]:
		predictions.append(predictor.predict(sentence))
	print("calculating weighted f1...")

	it = f1_weighted(predictions, allChunks)
	print(it)
	print("-----------------------------")
	weightedAverage += it
	print("calculating micro average f1...")

	alsoIt = f1_micro(predictions, allChunks)
	print(alsoIt)
	print("-----------------------------")
	microAverage += alsoIt
	print("calculating macro average f1...")
	lastIt = f1_macro(predictions, allChunks)
	print(lastIt)
	print("-----------------------------")
	macroAverage += lastIt

weightedAverage = weightedAverage / numberOfSlices
microAverage = microAverage / numberOfSlices
macroAverage = macroAverage / numberOfSlices

print("SSPRLG Weighted average: " + str(weightedAverage))
print("SSPRLG micro average: " + str(microAverage))
print("SSPRLG macro average: " + str(macroAverage))

