import numpy as np
from numpy import sin, cos, log, exp
from scipy.sparse import coo_matrix as cmat
import matplotlib.pyplot as plt
np.random.seed(21)

def softmax(z):
    e = exp(z - z.max(1, keepdims=True))
    a = e / e.sum(1, keepdims=True)
    return a

def relu(z): return np.maximum(0,z)

def to_onehot(y, C):
    data = np.ones_like(y)
    row, col = range(len(y)), y
    return cmat((data, (row, col)), shape=(len(y),C)).toarray()


n = 100 # nsample_per_class
d = 2 # xdim
C = 3 # nclass
X = np.zeros((n*C, d))
y = np.array([0]*n + [1]*n + [2]*n, dtype='int8')

N = X.shape[0]
Y = to_onehot(y, C)

for j in range(C):
    ids = range(n*j, n*(j+1))
    r = np.linspace(0., 1, n)
    t = np.linspace(4*j, 4*(j+1), n) + .2*np.random.randn(n)
    X[ids,:] = np.c_[r*sin(t), r*cos(t)]


def cost(Y, Q): return -(Y*log(Q)).sum()/Y.shape[0]

def predict(X):
    Z1 = X.dot(W1) + b1
    A1 = relu(Z1)
    Z2 = A1.dot(W2) + b2
    Yprob = softmax(Z2)
    ypred = np.argmax(Yprob, axis=1)
    return ypred

h = 20
d0, d1, d2 = d, h, C

W1 = .01 * np.random.randn(d0, d1)
b1 = np.zeros((1, d1))
W2 = .01 * np.random.randn(d1, d2)
b2 = np.zeros((1, d2))

lr = 2
epochs = 2000

# train
for epoch in range(epochs):
    # forward
    Z1 = X.dot(W1) + b1
    A1 = relu(Z1)
    Z2 = A1.dot(W2) + b2
    Q = softmax(Z2)

    if epoch % 500 == 0:
        loss = cost(Y, Q)
        print('.epoch {:5}: loss = {:.4f}'.format(epoch, loss))

    # backward
    dZ2 = (Q - Y) / N
    dW2 = A1.T.dot(dZ2)
    db2 = dZ2.sum(0, keepdims=True)

    dZ1 = dZ2.dot(W2.T)
    dZ1[Z1 <= 0] = 0
    dW1 = X.T.dot(dZ1)
    db1 = dZ1.sum(0, keepdims=True)

    # step
    W1 += -lr * dW1
    b1 += -lr * db1
    W2 += -lr * dW2
    b2 += -lr * db2

# test
ypred = predict(X)
accu  = np.mean(ypred == y)
print('\naccuracy: {:.5f}'.format(accu))

# visualize
xv = np.arange(-1.5, 1.5, .01)
yv = np.arange(-1.5, 1.5, .01)
xv, yv = np.meshgrid(xv, yv)

XV = np.vstack((xv.flatten(), yv.flatten())).T
zv = predict(XV).reshape(xv.shape)

fig, ax = plt.subplots(figsize=(6,6), dpi=100)
ax.contourf(xv, yv, zv, levels=3, cmap='Accent_r', alpha=.3)
ax.scatter(X[:,0], X[:,1], c=y, cmap='Accent', marker='.', edgecolor='face')
ax.set(xlim=[-1.5,1.5], ylim=[-1.5,1.5])
ax.axis('square')
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('MLP')
fig.savefig('mlp.jpg', dpi=100)
plt.show()