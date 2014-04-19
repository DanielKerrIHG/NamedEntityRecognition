from src.data import *
print("loading...")
loadData = importDetailed
train = loadData('conll2000/train.txt')
test = loadData('conll2000/test.txt')

combined0 = loadData('conll2000/combined.txt')
combined1 = loadData('conll2000/combined.txt')
combined2 = loadData('conll2000/combined.txt')

allPos = set(word.pos for sentence in train + test for word in sentence)
allChunks = set(word.chunk for sentence in train + test for word in sentence)

#HMM
print("Slicing and dicing data...")
testSlicesHMM = []
trainSlicesHMM = []
numberOfSlices = 10
sliceData(combined0, numberOfSlices, trainSlicesHMM, testSlicesHMM)

#PRLG
testSlicesPRLG = []
trainSlicesPRLG = []
numberOfSlices = 10
sliceData(combined1, numberOfSlices, trainSlicesPRLG, testSlicesPRLG)

#PLLG
testSlicesPLLG = []
trainSlicesPLLG = []
numberOfSlices = 10
sliceData(combined2, numberOfSlices, trainSlicesPLLG, testSlicesPLLG)


from src.hmm import *
from src.prlg import *
from src.pllg import *

print('===================HMM=====================')
print("calculating data slices...")
weightedAverage = 0
precisionTotal = 0
recallTotal = 0
for i in range(numberOfSlices):
	print("calculating using slice " + str(i) + " as testing data:")
	predictor = Markov(allPos, allChunks).train(trainSlicesHMM[i])
	predictions = []
	for sentence in testSlicesHMM[i]:
		predictions.append(predictor.predict(sentence))
	print("calculating weighted f1...")
	print("Precision: ")
	pre = precision_weighted(predictions, allChunks)
	print(pre)
	precisionTotal += pre
	print("Recall: ")
	rec = recall_weighted(predictions, allChunks)
	print(rec)
	recallTotal += rec
	print("Weighted F1: ")
	it = f1_weighted(predictions, allChunks)
	print(it)
	print("----------------------------------------")
	weightedAverage += it


weightedAverage = weightedAverage / numberOfSlices
precisionAverage = precisionTotal / numberOfSlices
recallAverage = recallTotal / numberOfSlices

print("Weighted average: " + str(weightedAverage))
print("Precision average: " + str(precisionAverage))
print("recall average: " + str(recallAverage))

print('===================PRLG=====================')
print("calculating data slices...")
weightedAverage = 0
precisionTotal = 0
recallTotal = 0
for i in range(numberOfSlices):
	print("calculating using slice " + str(i) + " as testing data:")
	predictor = PRLG(allPos, allChunks).train(trainSlicesPRLG[i])
	predictions = []
	for sentence in testSlicesPRLG[i]:
		predictions.append(predictor.predict(sentence))
	print("calculating weighted f1...")
	print("Precision: ")
	pre = precision_weighted(predictions, allChunks)
	print(pre)
	precisionTotal += pre
	print("Recall: ")
	rec = recall_weighted(predictions, allChunks)
	print(rec)
	recallTotal += rec
	print("Weighted F1: ")
	it = f1_weighted(predictions, allChunks)
	print(it)
	print("----------------------------------------")
	weightedAverage += it


weightedAverage = weightedAverage / numberOfSlices
precisionAverage = precisionTotal / numberOfSlices
recallAverage = recallTotal / numberOfSlices

print("Weighted average: " + str(weightedAverage))
print("Precision average: " + str(precisionAverage))
print("recall average: " + str(recallAverage))

print('===================PLLG=====================')
print("calculating data slices...")
weightedAverage = 0
precisionTotal = 0
recallTotal = 0
for i in range(numberOfSlices):
	print("calculating using slice " + str(i) + " as testing data:")
	predictor = PLLG(allPos, allChunks).train(trainSlicesPLLG[i])
	predictions = []
	for sentence in testSlicesPLLG[i]:
		predictions.append(predictor.predict(sentence))
	print("calculating weighted f1...")
	print("Precision: ")
	pre = precision_weighted(predictions, allChunks)
	print(pre)
	precisionTotal += pre
	print("Recall: ")
	rec = recall_weighted(predictions, allChunks)
	print(rec)
	recallTotal += rec
	print("Weighted F1: ")
	it = f1_weighted(predictions, allChunks)
	print(it)
	print("----------------------------------------")
	weightedAverage += it


weightedAverage = weightedAverage / numberOfSlices
precisionAverage = precisionTotal / numberOfSlices
recallAverage = recallTotal / numberOfSlices

print("Weighted average: " + str(weightedAverage))
print("Precision average: " + str(precisionAverage))
print("recall average: " + str(recallAverage))