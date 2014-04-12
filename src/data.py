class Word:
	def __init__(self, spelling, pos, chunk):
		self.spelling = spelling
		self.pos = pos
		self.chunk = chunk

# type Sentence = [Word]
# type Word = (str, POS, Chunk)
# type POS = str
# type Chunk = String

def importData(filepath):
	sentences = []
	with open(filepath, 'r') as f:
		acc = []
		for line in f.readlines():
			if len(line.strip()):
				acc.append(Word(*line.split()))
			elif acc:
				sentences.append(acc)
				acc = []
	return sentences


train = importData('conll2000/train.txt')
test = importData('conll2000/test.txt')


