import itertools
import csv
import nltk
import numpy as np
from rnn import RNN, train
np.random.seed(21)

nvocab = 500
TK_UNKN = 'TK_UNKN'
TK_START = 'TK_START'
TK_END = 'TK_END'

# read csv
with open('reddit1508.csv', 'r', encoding='utf8') as f:
    reader = csv.reader(f)
    sens = [r[0].lower() for r in reader]
print('sens: ', len(sens))

# sent_tokenize
sens = sens[:100]
sens = itertools.chain(*[nltk.sent_tokenize(s) for s in sens])
sens = ['{} {} {}'.format(TK_START, s, TK_END) for s in sens]
print('sens: ', len(sens))

# word_tokenize
sens1 = [nltk.word_tokenize(s) for s in sens]
word_freq = nltk.FreqDist(itertools.chain(*sens1))
print('word_freq: ', len(word_freq))

# sent_clean
vocab = word_freq.most_common(nvocab-1)
i2w = [x[0] for x in vocab]
i2w.append(TK_UNKN)
w2i = {v:k for k,v in enumerate(i2w)}

sens2 = [[w if w in i2w else TK_UNKN for w in s] for s in sens1]
sens2 = [[w2i[w] for w in s] for s in sens2]

# train data
X = np.array([s[:-1] for s in sens2])
Y = np.array([s[1:] for s in sens2])

print('X shape: ', X.shape)
print('Y shape: ', Y.shape)

# model
model = RNN(nvocab, 50, 4)
losses = train(model, X, Y, 0.002, 5, 1)

