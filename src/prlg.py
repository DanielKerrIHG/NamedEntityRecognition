from .hmm import Markov

class PRLG(Markov):
    def __init__(self, poss, chunks):
        self.poss = poss
        self.chunks = chunks
        self.initial = {chunk: 0 for chunk in chunks}
        self.transition = {chunk1: {chunk2: 0 for chunk2 in chunks} for chunk1 in chunks}
        self.emission = {(chunk1, chunk2): {pos: 0 for pos in poss} for chunk1 in chunks for chunk2 in chunks}
        self.emission.update({(chunk1, None): {pos: 0 for pos in poss} for chunk1 in chunks})

    def _trainEmission(self, data):
        # count the associations of a chunk bigram with an observable
        for sentence in data:
            for t in range (0, len(sentence)-1):
                self.emission[(sentence[t].chunk, sentence[t+1].chunk)][sentence[t].pos] += 1
            t = len(sentence)-1
            self.emission[(sentence[t].chunk, None)][sentence[t].pos] += 1
        # each row must som to 1
        for bigram, row in self.emission.items():
            total = sum(row.values())
            if total != 0:
                for pos in row:
                    self.emission[bigram][pos] /= total

    def predict(self, sentence):
        raise NotImplementedError("TODO")