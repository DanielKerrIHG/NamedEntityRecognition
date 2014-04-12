class Word:
	def __init__(self, spelling, pos, chunk):
		self.spelling = spelling
		self.pos = pos
		self.chunk = chunk

# type Sentence = [Word]
# type Word = (str, POS, Chunk)
# type POS = str
# type Chunk = String

def importDetailed(filepath):
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

def importSimple(filepath):
	sentences = importDetailed(filepath)
	for sentence in sentences:
		for word in sentence:
			word.chunk = word.chunk[0]
	return sentences

train = importSimple('conll2000/train.txt')
test = importSimple('conll2000/test.txt')


