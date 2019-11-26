import numpy as np
from sklearn.naive_bayes import BernoulliNB

class Word2Vec(object):
    def __init__(self, sentences):
        # self.words = self._get_words(sentences)
        self.words = 'hanoi pho chaolong buncha omai banhgio saigon hutiu banhbo'.split()
        self.w2v = {v: k for (k, v) in enumerate(self.words)}

    def _get_words(self, sentences):
        words = []
        for s in sentences:
            words.extend(s.split())
        words = sorted(set(words))
        return words

    def _vectorize(self, sentence):
        vec = [self.w2v[x] for x in sentence.split()]
        vec = [1 if vec.count(j) > 0 else 0 for j in range(len(self.w2v))]
        return vec

    def transform(self, sentences):
        vecs = [self._vectorize(s) for s in sentences]
        return vecs


d1 = 'hanoi pho chaolong hanoi'
d2 = 'hanoi buncha pho omai'
d3 = 'pho banhgio omai'
d4 = 'saigon hutiu banhbo pho'
d5 = 'hanoi hanoi buncha hutiu'
d6 = 'pho hutiu banhbo'

sentences = [d1,d2,d3,d4,d5,d6]
W2V = Word2Vec(sentences)
X = W2V.transform(sentences)

X_trn = np.array(X[:4])
y_trn = np.array(['B','B','B','N'])

NB = BernoulliNB()
NB.fit(X_trn, y_trn)

# test
X_tst = [X[4]]
pred = NB.predict(X_tst)
print(pred)

X_tst = [X[5]]
probs = NB.predict_proba(X_tst)
print(probs)