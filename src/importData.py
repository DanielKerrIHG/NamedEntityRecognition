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
				word, pos, chunk = line.split()
				acc.append((word, pos, chunk))
			elif acc:
				sentences.append(acc)
				acc = []
	return sentences