import numpy as np
from numpy import log, exp
from scipy import sparse
import matplotlib.pyplot as plt
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

# DATA
means = [[2,2], [8,3], [3,6]]
cov = [[1,0], [0,1]]
N = 500
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)
X = np.vstack((X0,X1,X2))
print('origin: ', X.shape)

C = 3
one = np.ones((3*N, 1))
X1 = np.hstack((one, X))
y = np.array([0]*N + [1]*N + [2]*N)
print('train: ', X1.shape, y.shape)

# MODEL
def pred(X, W):
    A = softmax(X.dot(W))
    return np.argmax(A, axis=1)

def grad(X, Y, W):
    A = softmax(X.dot(W))
    E = A - Y
    return X.T.dot(E)

# OPTIM
def gd(X, y, W0, lr, eps=1e-4, epochs=1000, iter_check=20, batch_size=100):
    Ws = [W0]
    N = X.shape[0]
    d, C = W0.shape
    Y = to_onehot(y, C)

    it = 0
    for epoch in range(epochs):
        ids = np.random.permutation(N)
        for j in range(0, len(ids), batch_size):
            js = ids[j: j+batch_size]
            Xi = X[js]
            Yi = Y[js]
            W_new = Ws[-1] - lr * grad(Xi, Yi, Ws[-1])
            it += 1

            # stop criteria
            if it % iter_check == 0:
                if np.linalg.norm(W_new - Ws[-iter_check]) < eps:
                    return Ws
            Ws.append(W_new)
    print('{} iters in {} epochs'.format(it, epoch))
    return Ws

lr = .5
W0 = np.random.randn(X1.shape[1], C)
Ws = gd(X1, y, W0, lr, eps=1e-3, batch_size=200)
W = Ws[-1]

# VISUALIZE
xm = np.arange(-2,11, .025)
ym = np.arange(-3,10, .025)
xx, yy = np.meshgrid(xm,ym)

xx1 = xx.reshape(-1, 1)
yy1 = yy.reshape(-1, 1)
one = np.ones((xx1.shape[0], 1))

XX = np.hstack((one, xx1, yy1))
zz = pred(XX, W)
zz = zz.reshape(xx.shape)
print('XX ({}), zz ({})'.format(XX.shape, zz.shape))

plt.figure()
plt.contourf(xx,yy,zz, cmap='jet', alpha=.1)
plt.scatter(X[:,0], X[:,1], c=y, cmap='Set1', marker='.', alpha=.7)
plt.grid(alpha=.1)
plt.show()