from __future__ import division, print_function, unicode_literals
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
np.random.seed(21)

# data
N = 1000
X = np.random.rand(N, 1)
y = 4 + 3*X + .2*np.random.randn(N, 1)
one = np.ones((N, 1))
X1 = np.hstack((one, X))


def func(w): return .5/N * norm(y - X1.dot(w), 2)**2


def grad(w): return 1./N * X1.T.dot(X1.dot(w) - y)


def sgrad(w, xi, yi): return xi.T*(xi.dot(w) - yi)


def mgrad(w, Xi, yi): return 1./Xi.shape[0] * Xi.T.dot(Xi.dot(w) - yi)


def sgd1(X, y, w_init, lr=0.1, n_epochs=10, eps=1e-3, it_check=10):
    N = X.shape[0]
    w = w_init
    w_check = w.copy()
    it = 0
    for epoch in range(n_epochs):
        shuffle_id = np.random.permutation(N)
        for j in shuffle_id:
            it += 1
            xi = X[j].reshape(-1, X.shape[1])
            yi = y[j].reshape(-1, y.shape[1])
            g = sgrad(w, xi, yi)
            w = w - lr * g
            if it % it_check == 0:
                if norm(w - w_check)/len(w) < eps:
                    return w, it
                w_check = w.copy()
    return w, it


def mgd1(X, y, w_init, lr=0.1, n_epochs=10, eps=1e-3, it_check=10, batch_size=20):
    N = X.shape[0]
    w = w_init
    w_check = w.copy()
    it = 0
    for epoch in range(n_epochs):
        shuffle_id = np.random.permutation(N)
        for j in range(0, N, batch_size):
            it += 1
            batch_id = shuffle_id[j: j+batch_size]
            Xi = X[batch_id].reshape(-1, X.shape[1])
            yi = y[batch_id].reshape(-1, y.shape[1])
            g = mgrad(w, Xi, yi)
            w = w - lr * g
            if it % it_check == 0:
                if norm(w - w_check)/len(w) < eps:
                    return w, it
                w_check = w.copy()
    return w, it


# gd
w0 = np.array([[2], [1]])
w, it = sgd1(X1, y, w0, .1)
print('sgd has stoped after {} iters'.format(it))
print('f = {:4.2f}*x + {:4.2f}'.format(w[1][0], w[0][0]))

w, it = mgd1(X1, y, w0, .1)
print('mgd has stoped after {} iters'.format(it))
print('f = {:4.2f}*x + {:4.2f}'.format(w[1][0], w[0][0]))
