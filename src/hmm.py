class Markov:
    def __init__(self, initial, transition, emission, poss, chunks):
        self.initial = initial
        self.transition = transition
        self.emission = emission
        self.poss = poss
        self.chunks = chunks

    @classmethod
    def empty(cls, poss, chunks):
        init = {chunk: 0 for chunk in chunks}
        sttm = {chunk1: {chunk2: 0 for chunk2 in chunks} for chunk1 in chunks}
        emm = {chunk: {pos: 0 for pos in poss} for chunk in chunks}
        return cls(init, sttm, emm, poss, chunks)

    def train(self, data):
        ## First, make a raw count of first words, bigrams and obs-hidden associations
        for sentence in data:
            # count the chunk value on the first word of every sentence
            self.initial[sentence[0].chunk] += 1
            # count the bigrams of the chunk property
            for t in range(1, len(sentence)):
                self.transition[sentence[t-1].chunk][sentence[t].chunk] += 1
            # count how often a POS appears in conjunction with a chunk
            for t in range(0, len(sentence)):
                self.emission[sentence[t].chunk][sentence[t].pos] += 1
        ## Then, ensure the probabilities all sum to 1 where appropriate
        # the vector must sum to 1
        total = sum(self.initial.values())
        for st in self.initial:
            self.initial[st] /= total
        # each row must sum to 1
        for st1, row in self.transition.items():
            total = sum(row.values())
            if total != 0:
                for st2 in row:
                    self.transition[st1][st2] /= total
        # each row must sum to 1
        for st, row in self.emission.items():
            total = sum(row.values())
            if total != 0:
                for pos in row:
                    self.emission[st][pos] /= total
        return self
    
    def viterbi(self, sentence):
        V = [{chunk: self.initial[chunk] * self.emission[chunk][sentence[0].pos] for chunk in self.chunks}]
        path = {chunk: [chunk] for chunk in self.chunks}

        def step_prob(chunk1, chunk2, t):
            return V[t-1][chunk1] * self.transition[chunk1][chunk2] * self.emission[chunk2][sentence[t].pos]

        for t in range(1, len(sentence)):
            V.append(dict())
            newpath = dict()

            for chunk2 in self.chunks:
                p, likelyChunk = max((step_prob(chunk1, chunk2, t), chunk1) for chunk1 in self.chunks)
                V[t][chunk2] = p
                newpath[chunk2] = path[likelyChunk] + [chunk2]
            path = newpath

        assert t == len(sentence) - 1
        p, likely = max((V[t][chunk], chunk) for chunk in self.chunks)

        for i, word in enumerate(sentence):
            word.predicted = path[likely][i]
        return sentence


def precision(sentences, chunk):
    truePositive, falsePositive = 0, 0
    for sentence in sentences:
        for word in sentence:
            if chunk == word.predeicted:
                if word.predicted == word.chunk:
                    truePositive += 1
                else:
                    falsePositive += 1
    return truePostive / (truePostive + falsePositive)

def recall(sentences, chunk):
    truePostive, falseNegative = 0, 0
    for sentence in sentences:
        or word in sentence:
            if chunk == word.predeicted:
                if word.predicted == word.chunk:
                    truePositive += 1
            if chunk == word.chunk:
                if word.predicted != word.chunk:
                    falseNegative += 1
    return truePositive / (truePositive + falseNegative)

def f1_measure(sentences, chunk):
    p = precision(sentences, chunk)
    r = recall(sentences, chunk)
    return 2 * (p * r)/(p + r)



