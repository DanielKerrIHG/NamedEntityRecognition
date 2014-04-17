class PLLG:
    def __init__(self, poss, chunks):
        self.poss = poss
        self.chunks = chunks
        self.initial = {chunk: 0 for chunk in chunks}
        self.back_trans = {chunk2: {chunk1: 0 for chunk1 in chunks} for chunk2 in chunks}
        self.emission = {chunk: {pos: 0 for pos in poss} for chunk in chunks}
        self.biemission = {(chunk1, chunk2): {pos: 0 for pos in poss} for chunk1 in chunks for chunk2 in chunks}

    def train(self, data):
        self._trainInitial(data)
        self._trainBackTrans(data)
        self._trainEmission(data)
        self._trainBiemission(data)
        return self


    def _trainInitial(self, data):
        # count the chunk value on the first word of every sentence
        for sentence in data:
            self.initial[sentence[0].chunk] += 1
        # the vector must sum to 1
        total = sum(self.initial.values())
        for st in self.initial:
            self.initial[st] /= total

    def _trainBackTrans(self, data):
        # count the bigrams of the chunk property
        for sentence in data:
            for t in range(1, len(sentence)):
                self.back_trans[sentence[t].chunk][sentence[t-1].chunk] += 1
        # each row must sum to 1
        for st1, row in self.back_trans.items():
            total = sum(row.values())
            if total != 0:
                for st2 in row:
                    self.back_trans[st1][st2] /= total

    def _trainEmission(self, data):
        for sentence in data:
            # count how often a POS appears in conjunction with a chunk
            for t in range(0, len(sentence)):
                self.emission[sentence[t].chunk][sentence[t].pos] += 1
        # each row must sum to 1
        for st, row in self.emission.items():
            total = sum(row.values())
            if total != 0:
                for pos in row:
                    self.emission[st][pos] /= total

    def _trainBiemission(self, data):
        for sentence in data:
            # count how often a POS appears in conjunction with a chunk
            for t in range(1, len(sentence)):
                self.biemission[(sentence[t-1].chunk, sentence[t].chunk)][sentence[t].pos] += 1
        # each row must sum to 1
        for st, row in self.biemission.items():
            total = sum(row.values())
            if total != 0:
                for pos in row:
                    self.biemission[st][pos] /= total

    
    def predict(self, sentence):
        V = [{chunk: self.initial[chunk] * self.emission[chunk][sentence[0].pos] for chunk in self.chunks}]
        path = {chunk: [chunk] for chunk in self.chunks}

        def step_prob(chunk1, chunk2, t):
            return V[t-1][chunk1] * self.back_trans[chunk2][chunk1] * self.biemission[(chunk1,chunk2)][sentence[t].pos]

        t = 0
        for t in range(1, len(sentence)):
            V.append(dict())
            newpath = dict()

            for chunk2 in self.chunks:
                p, likelyChunk = max((step_prob(chunk1, chunk2, t), chunk1) for chunk1 in self.chunks)
                V[t][chunk2] = p
                newpath[chunk2] = path[likelyChunk] + [chunk2]
            path = newpath

        try:
            assert t == len(sentence) - 1
        except:
            raise Exception(sentence)
        p, likely = max((V[t][chunk], chunk) for chunk in self.chunks)

        for i, word in enumerate(sentence):
            word.predicted = path[likely][i]
        return sentence

