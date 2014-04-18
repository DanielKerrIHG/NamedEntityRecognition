"""Steps:
	1) seperate data into labeledData, unlabledData, testData
	2) train HMM on labeledData as normal
		while predictions change:
			3) predict unlabledData using trained HMM
			4) set label of unlabledData to be that of prediction
			5) train HMM using unlabledData
			
	6) return converged trainer"""

from .data import *
from .hmm import *

def semiTrain(data, allPos, allChunks):
	# Step 1)
	labeledSlices = []
	unlabledSlices = []
	c = 0
	for sentence in data:
		if c % 200 == 0:
			labeledSlices.append(sentence)
		else:
			unlabledSlices.append(sentence)
		c += 1
	
	# Step 2)
	trainedPredictor = Markov(allPos, allChunks).train(labeledSlices)
	c = 0
	changedCount = 1
	while changedCount > 0:
		changedCount = 0
		# Step 3)
		for sentence in unlabledSlices:
			tempSentence = trainedPredictor.predict(sentence)
			for i in range(len(sentence)):
				if tempSentence[i].predicted != sentence[i].chunk:
					changedCount += 1
				# Step 4)
				sentence[i].chunk = tempSentence[i].predicted
		# Step 5)
		trainedPredictor = Markov(allPos, allChunks).train(labeledSlices + unlabledSlices)
		
		c += 1
	# step 6)
	return trainedPredictor