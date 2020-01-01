import numpy as np
from numpy import log, exp
from scipy import sparse
from sklearn.metrics import accuracy_score as acc
np.random.seed(21)

def softmax(z):
    e = exp(z - z.max(1, keepdims=True))
    a = e / e.sum(1, keepdims=True)
    return a

def to_onehot(y, C):
    data = np.ones_like(y)
    row, col = range(len(y)), y
    Y = sparse.coo_matrix((data, (row, col)), shape=(len(y), C)).toarray()
    return Y

# RANDOM DATA
n_samples = 10
n_dim = 3
n_cls = 4

X = np.random.randn(n_samples, n_dim)
y = np.random.randint(0,3, (n_samples))
Y = to_onehot(y, n_cls)

# MODEL
def pred(X, W):
    A = softmax(X.dot(W))
    return np.argmax(A, axis=1)

def cost(X, Y, W):
    A = softmax(X.dot(W))
    return -np.sum(Y*log(A))

def grad(X, Y, W):
    A = softmax(X.dot(W))
    E = A - Y
    return X.T.dot(E)

def grad_num(X, Y, W, cost, eps=1e-6):
    g = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            Wp = W.copy()
            Wn = W.copy()
            Wp[i,j] += eps
            Wn[i,j] -= eps
            g[i,j] = (cost(X,Y,Wp) - cost(X,Y,Wn)) / (2*eps)
    return g

## CHECK GRAD
W0 = np.random.randn(n_dim, n_cls)
g1 = grad(X, Y, W0)
g2 = grad_num(X, Y, W0, cost)
print('diff grad: ', np.linalg.norm(g1 - g2))

## OPTIM
def gd(X, y, W0, lr, eps=1e-4, max_iter=10000, iter_check=20):
    Ws = [W0]
    N = X.shape[0]
    d, C = W0.shape
    Y = to_onehot(y, C)

    it = 0
    while it < max_iter:
        ids = np.random.permutation(N)
        for j in ids:
            xi = X[j].reshape(1,-1)
            yi = Y[j].reshape(1,-1)
            W_new = Ws[-1] - lr * grad(xi, yi, Ws[-1])
            it += 1

            # stop criteria
            if it % iter_check == 0:
                if np.linalg.norm(W_new - Ws[-iter_check]) < eps:
                    return Ws
            Ws.append(W_new)
    return Ws

Ws = gd(X, y, W0, lr=.05)
y_prd = pred(X, Ws[-1])

print(y)
print(y_prd)
accuracy = acc(y, y_prd)
print('accuracy: ', accuracy)