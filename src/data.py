class Word:
	def __init__(self, spelling, pos, chunk):
		self.spelling = spelling
		self.pos = pos
		self.chunk = chunk

	def __str__(self): return self.__repr__()
	def __repr__(self):
		if hasattr(self, 'predicted'):
			return str((self.spelling, self.pos, self.chunk, self.predicted))
		else:
			return str((self.spelling, self.pos, self.chunk))

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



