from .hmm import Markov

class PRLG(Markov):
    def __init__(self, poss, chunks):
        self.poss = poss
        self.chunks = chunks
        self.initial = {chunk: 0 for chunk in chunks}
        self.transition = {chunk1: {chunk2: 0 for chunk2 in chunks} for chunk1 in chunks}
        self.emission = {(chunk1, chunk2): {pos: 0 for pos in poss} for chunk1 in chunks for chunk2 in chunks}
        self.emission.update({(chunk1, None): {pos: 0 for pos in poss} for chunk1 in chunks})

    def _trainInitial(self, data):
        # count the chunk value on the first word of every sentence
        for sentence in data:
            self.initial[sentence[-1].chunk] += 1
        # the vector must sum to 1
        total = sum(self.initial.values())
        for st in self.initial:
            self.initial[st] /= total
    
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
        V = [dict() for x in range(0, len(sentence))]
        V[-1] = {chunk: self.initial[chunk] * self.emission[(chunk, None)][sentence[-1].pos] for chunk in self.chunks}
        path = {chunk: [chunk] for chunk in self.chunks}

        def _step(t, chunk1, chunk2):
            return V[t+1][chunk2] * self.transition[chunk1][chunk2] * self.emission[(chunk1, chunk2)][sentence[t].pos]

        t = 0
        for t in reversed(range(0, len(sentence) - 1)):
            newpath = dict()

            for chunk2 in self.chunks:
                p, likelyChunk = max((_step(t, chunk1, chunk2), chunk1) for chunk1 in self.chunks)
                V[t][chunk2] = p
                newpath[chunk2] = path[likelyChunk] + [chunk2]
            path = newpath

        try:
            assert t == 0
        except:
            raise Exception(sentence)
        p, likely = max((V[t][chunk], chunk) for chunk in self.chunks)

        for i, word in enumerate(sentence):
            word.predicted = path[likely][i]
        return sentence

