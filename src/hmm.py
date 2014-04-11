from .data import train, test


allPOS = set(pos for s in train + test for word, pos, chunk in s)
allChunk = set(chunk for s in train + test for word, pos, chunk in s)


def initialStateVector():
    return {chunk: 0 for chunk in allChunk}

def stateTransitionMatrix():
    return {chunk1: {chunk2: 0 for chunk2 in allChunk} for chunk1 in allChunk}

def emissionMatrix():
    return {chunk: {pos: 0 for pos in allPOS} for chunk in allChunk}


def count():
    init, stt, emm = initialStateVector(), stateTransitionMatrix(), emissionMatrix()
    for s in train:
        # count the chunk value on the first word of every sentence
        init[s[0][2]] += 1
        # count the bigrams of the chunk property
        for i in range(1, len(s)):
            stt[s[i-1][2]][s[i][2]] += 1
        # count how often a POS appears in conjunction with a chunk
        for i in range(0, len(s)):
            emm[s[i][2]][s[i][1]] += 1
    return init, stt, emm

def normalize(init, stt, emm):
    # the vector must sum to 1
    total = sum(init.values())
    for st in init:
        init[st] /= total
    # each row must sum to 1
    for st1, row in stt.items():
        total = sum(row.values())
        if total != 0:
            for st2 in row:
                stt[st1][st2] /= total
    # each row must sum to 1
    for st, row in emm.items():
        total = sum(row.values())
        if total != 0:
            for pos in row:
                emm[st][pos] /= total
    return init, stt, emm