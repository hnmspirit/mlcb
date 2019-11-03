import numpy as np
import matplotlib.pyplot as plt
np.random.seed(21)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation as fanim
np.random.seed(21)


N = 520
X = np.random.rand(N, 1)
y = 4 + 3*X + 0.2*np.random.randn(N, 1)

X1 = np.hstack((np.ones((N, 1)), X))
w0 = np.random.rand(2, 1)
print(w0[:,0])

def func(w): return .5/N * np.linalg.norm(y - X1.dot(w), 2)**2
def grad(w): return 1./N * X1.T.dot(X1.dot(w) - y)
def sgrad(w, xi, yi): return xi.T.dot(xi.dot(w) - yi)
def mgrad(w, Xi, yi): return (Xi.T.dot(Xi.dot(w) - yi)) / Xi.shape[0]


def gd(w_init, lr=0.1, epochs=1000, eps=1e-3):
    w = w_init.copy()
    for it in range(epochs):
        gw = grad(w)
        w -= lr * gw
        if np.linalg.norm(gw) / gw.size < eps:
            break
    return w, it


def sgd(w_init, lr=0.1, epochs=10, eps=1e-3, it_check=10):
    w = w_init.copy()
    w_check = w.copy()
    it = 0
    for epoch in range(epochs):
        ids = np.random.permutation(N)
        for j in ids:
            it += 1
            xi = X1[j:j+1]
            yi = y[j:j+1]
            g = sgrad(w, xi, yi)
            w = w - lr * g
            if it % it_check == 0:
                if np.linalg.norm(w - w_check) / w.size < eps:
                    return w, it
                w_check = w.copy()
    return w, it


def mgd(w_init, lr=0.1, epochs=10, eps=1e-3, it_check=10, batch_size=20):
    w = w_init.copy()
    w_check = w.copy()
    it = 0
    for epoch in range(epochs):
        ids = np.random.permutation(N)
        for j in range(0, N, batch_size):
            it += 1
            js = ids[j: j+batch_size]
            Xi = X1[js]
            yi = y[js]
            g = mgrad(w, Xi, yi)
            w = w - lr * g
            if it % it_check == 0:
                if np.linalg.norm(w - w_check) / w.size < eps:
                    return w, it
                w_check = w.copy()
    return w, it



w, it = gd(w0, 0.1, epochs=1000)
print('bgd: it=%5d, w= %.4f: %.4f' % (it+1, w[0, 0], w[1, 0]))

w, it = sgd(w0, 0.1, epochs=100)
print('sgd: it=%5d, w= %.4f: %.4f' % (it+1, w[0, 0], w[1, 0]))

w, it = mgd(w0, 0.1, epochs=100)
print('mgd: it=%5d, w= %.4f: %.4f' % (it+1, w[0, 0], w[1, 0]))