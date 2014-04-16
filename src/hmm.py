class Markov:
    def __init__(self, poss, chunks):
        self.poss = poss
        self.chunks = chunks
        self.initial = {chunk: 0 for chunk in chunks}
        self.transition = {chunk1: {chunk2: 0 for chunk2 in chunks} for chunk1 in chunks}
        self.emission = {chunk: {pos: 0 for pos in poss} for chunk in chunks}

    def train(self, data):
        self._trainInitial(data)
        self._trainTransition(data)
        self._trainEmission(data)
        return self


    def _trainInitial(self, data):
        # count the chunk value on the first word of every sentence
        for sentence in data:
            self.initial[sentence[0].chunk] += 1
        # the vector must sum to 1
        total = sum(self.initial.values())
        for st in self.initial:
            self.initial[st] /= total

    def _trainTransition(self, data):
        # count the bigrams of the chunk property
        for sentence in data:
            for t in range(1, len(sentence)):
                self.transition[sentence[t-1].chunk][sentence[t].chunk] += 1
        # each row must sum to 1
        for st1, row in self.transition.items():
            total = sum(row.values())
            if total != 0:
                for st2 in row:
                    self.transition[st1][st2] /= total

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

    
    def predict(self, sentence):
        V = [{chunk: self.initial[chunk] * self.emission[chunk][sentence[0].pos] for chunk in self.chunks}]
        path = {chunk: [chunk] for chunk in self.chunks}

        def step_prob(chunk1, chunk2, t):
            return V[t-1][chunk1] * self.transition[chunk1][chunk2] * self.emission[chunk2][sentence[t].pos]

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


def precision(sentences, chunk):
    truePositive, falsePositive = 0, 0
    for sentence in sentences:
        for word in sentence:
            if chunk == word.predicted:
                if word.predicted == word.chunk:
                    truePositive += 1
                else:
                    falsePositive += 1
    if truePositive == 0 and falsePositive == 0:
        return 0
    else:
        return truePositive / (truePositive + falsePositive)


def recall(sentences, chunk):
    truePositive, falseNegative = 0, 0
    for sentence in sentences:
        for word in sentence:
            if chunk == word.predicted:
                if word.predicted == word.chunk:
                    truePositive += 1
            if chunk == word.chunk:
                if word.predicted != word.chunk:
                    falseNegative += 1
    if truePositive == 0 and falseNegative == 0:
        return 0
    else:
        return truePositive / (truePositive + falseNegative)

def f1_measure(sentences, chunk):
    p = precision(sentences, chunk)
    r = recall(sentences, chunk)
    if p == 0 and r == 0:
        return 0
    else:
        return 2 * (p * r)/(p + r)

def f1_weighted(predictions, chunks):
    totalF1 = 0
    totalCount = 0
    for chunk in chunks:
        localCount = 0
        for sentence in predictions:
            for word in sentence:
                if word.chunk == chunk:
                    localCount += 1
        if localCount != 0:
            totalCount += localCount
            totalF1 += localCount * f1_measure(predictions, chunk)
    return totalF1 / totalCount

def f1_micro(predictions, chunks):
    tp, fn, fp = 0, 0, 0
    for chunk in chunks:
        for sentence in predictions:
            for word in sentence:
                if chunk == word.predicted:
                    if word.predicted == word.chunk:
                        tp += 1
                    else:
                        fp += 1
                if chunk == word.chunk:
                    if word.predicted != word.chunk:
                        fn += 1
    precis = tp / (tp + fp)
    rec = tp / (tp + fn)
    return 2 * (precis * rec) / (precis + rec)

def f1_macro(predictions, chunks):
    runningTotal, chunkCount = 0, 0
    for chunk in chunks:
        runningTotal += f1_measure(predictions, chunk)
        chunkCount += 1
    return runningTotal / chunkCount






