train = importSimple('conll2000/train.txt')
test = importSimple('conll2000/test.txt')
detailedPOS = set(word.pos for sentence in train + test for word in sentence)
detailedChunks = set(word.chunk for sentence in train + test for word in sentence)

print("loaded")
from src.hmm import *


it = Markov.empty(detailedPOS, detailedChunks).train(train).viterbi(test[0])

