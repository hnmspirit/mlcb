import numpy as np
import matplotlib.pyplot as plt


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
        vec = [vec.count(j) for j in range(len(self.w2v))]
        return vec

    def transform(self, sentences):
        vecs = [self._vectorize(s) for s in sentences]
        return vecs


d0 = 'hanoi pho chaolong hanoi'
d1 = 'hanoi buncha pho omai'
d2 = 'pho banhgio omai'
d3 = 'saigon hutiu banhbo pho'
d4 = 'hanoi hanoi buncha hutiu'
d5 = 'pho hutiu banhbo'

labels = ['Bac', 'Nam']
y_all = [0,0,0,1,0,1]

sentences = [d0,d1,d2,d3,d4,d5]
W2V = Word2Vec(sentences)
X = W2V.transform(sentences)

X_trn = np.array(X[:4])
y_trn = y_all[:4]


# naive bayes
class NaiveBayes(object):
    def __init__(self, X_trn, y_trn):
        self.N, self.d = X_trn.shape
        self.classes = np.unique(y_trn)
        self.prios = np.array([np.sum(y_trn == c) for c in self.classes])

        totals = np.array([X_trn[y_trn == c].sum(axis=0) for c in self.classes])
        self.lambdas = (totals + 1) / (totals.sum(axis=1, keepdims=True) + self.d)

    def predict_proba(self, x):
        probs = self.prios / self.N * np.prod(self.lambdas ** x, axis=1)
        probs /= probs.sum()
        return probs

    def predict(self, x):
        probs = self.predict_proba(x)
        pred = np.argmax(probs)
        return pred

NB = NaiveBayes(X_trn, y_trn)

# test
x_tst = X[4]
pred = NB.predict(x_tst)
print(labels[pred])

x_tst = X[5]
probs = NB.predict_proba(x_tst)
pred = probs.argmax()
print(labels[pred])