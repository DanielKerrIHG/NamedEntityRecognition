from src.data import *
train = importDetailed('conll2000/train.txt')
test = importDetailed('conll2000/test.txt')
detailedPOS = set(word.pos for sentence in train + test for word in sentence)
detailedChunks = set(word.chunk for sentence in train + test for word in sentence)

print("loaded")
from src.hmm import *


predictor = Markov.empty(detailedPOS, detailedChunks).train(train)
predictions = []
for sentence in test:
	predictions.append(predictor.viterbi(sentence))
it = f1_measure(predictions, 'B-NP')


