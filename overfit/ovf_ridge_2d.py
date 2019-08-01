import numpy as np
from numpy import exp, log
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy import sparse
np.random.seed(4)

means = np.array([[-1, -1], [1, -1], [0, 1]])
cov = [[1, 0], [0, 1]]
N = 20
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

K = 3
X = np.vstack((X0, X1, X2)).T
y = np.array([0] * N + [1] * N + [2] * N)


def softmax(z):
    ez = exp(z - z.max(0, keepdims=True))
    az = ez / ez.sum(0, keepdims=True)
    return az


def to_onehot(y, C=3):
    data = np.ones_like(y)
    row, col = y, range(len(y))
    Y = sparse.coo_matrix((data, (row, col)), shape=(C, len(y))).toarray()
    return Y


gamma = 0.001


def cost(Y, Yhat, W1, W2, gamma):
    return -np.sum(Y * log(Yhat)) / Y.shape[1] + \
        gamma * (norm(W1)**2 + norm(W2)**2)


def predict(X, W1, b1, W2, b2):
    Z1 = W1.T.dot(X) + b1
    A1 = np.maximum(0, Z1)
    Z2 = W2.T.dot(A1) + b2
    Yhat = softmax(Z2)
    y_prd = np.argmax(Yhat, axis=0)
    return y_prd


d0 = 2
d1 = h = 100
d2 = C = 3


def mynet(gamma):
    W1 = 0.01 * np.random.randn(d0, d1)
    b1 = np.zeros((d1, 1))
    W2 = 0.01 * np.random.randn(d1, d2)
    b2 = np.zeros((d2, 1))

    Y = to_onehot(y, C)
    N = X.shape[1]
    lr = 1
    for i in range(10000):
        Z1 = W1.T.dot(X) + b1
        A1 = np.maximum(0, Z1)
        Z2 = W2.T.dot(A1) + b2
        Yhat = softmax(Z2)

        if i % 1000 == 0:
            loss = cost(Y, Yhat, W1, W2, gamma)
            print('it {}, loss={:f} '.format(i, loss))

        E2 = (Yhat - Y) / N
        dW2 = A1.dot(E2.T) + gamma * W2
        db2 = np.sum(E2, axis=1, keepdims=True)

        E1 = W2.dot(E2)
        E1[Z1 <= 0] = 0
        dW1 = X.dot(E1.T) + gamma * W1
        db1 = np.sum(E1, axis=1, keepdims=True)

        W1 += -lr * dW1
        b1 += -lr * db1
        W2 += -lr * dW2
        b2 += -lr * db2

    # test
    y_prd = predict(X, W1, b1, W2, b2)
    accu = 100 * np.mean(y_prd == y)
    print('train acc: {:.2f}%'.format(accu))

    xm = np.arange(-4, 4, 0.025)
    ym = np.arange(-4, 4, 0.025)
    xx, yy = np.meshgrid(xm, ym)

    XX = np.vstack((xx.ravel(), yy.ravel()))
    ZZ = predict(XX, W1, b1, W2, b2).reshape(xx.shape)

    plt.contourf(xx, yy, ZZ, 200, cmap='jet', alpha=.1)
    plt.scatter(X[0], X[1], c=y, cmap='Set1', marker='o')
    plt.xticks([]), plt.yticks([])
    plt.grid(alpha=.1), plt.title('$\lambda=${}'.format(gamma))


plt.figure(figsize=(8, 5))
plt.subplot(221)
mynet(gamma=0)
plt.subplot(222)
mynet(gamma=0.001)
plt.subplot(223)
mynet(gamma=0.01)
plt.subplot(224)
mynet(gamma=0.1)
plt.show()
