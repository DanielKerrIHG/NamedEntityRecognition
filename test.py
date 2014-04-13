from src.data import *
train = importDetailed('conll2000/train.txt')
test = importDetailed('conll2000/test.txt')
detailedPOS = set(word.pos for sentence in train + test for word in sentence)
detailedChunks = set(word.chunk for sentence in train + test for word in sentence)

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





