import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
np.random.seed(21)

# data
means = [[2, 2], [4, 3]]
cov = [[.7, 0], [0, .7]]
N = 20
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X = np.vstack([X0, X1])
y = np.array([1]*N + [-1]*N)

X0_bar = np.vstack((X0.T, np.ones((1, N))))
X1_bar = np.vstack((X1.T, np.ones((1, N))))

# model
C = 100
lam = 1./C
Z = np.hstack((X0_bar, -X1_bar))

def cost(w):
    # hinge loss
    u = w.T.dot(Z) # (d,C) @ (d,N)
    reg = .5*lam*np.sum(w[:-1]**2) # no bias
    return np.sum(np.maximum(0, 1-u)) + reg

def grad(w):
    u = w.T.dot(Z)
    H = np.where(u < 1)[1]
    ZS = Z[:, H]
    g = -np.sum(ZS, axis=1, keepdims=True) + lam*w
    g[-1] -= lam*w[-1]  # no bias
    return g

eps = 1e-6

# check_grad
def num_grad(w):
    g = np.zeros_like(w)
    for i in range(len(w)):
        wp = w.copy()
        wn = w.copy()
        wp[i] += eps
        wn[i] -= eps
        g[i] = (cost(wp) - cost(wn)) / (2*eps)
    return g


w0 = np.random.randn(X0_bar.shape[0], 1)
g1 = grad(w0)
g2 = num_grad(w0)
diff = np.linalg.norm(g1 - g2)
print('grad diff: %f' % diff)

# fit
def gd(w0, eta):
    w = w0
    it = 0
    while it < 100000:
        if it % 20000 == 0:
            print('it {:>5}, cost: {:.4f}'.format(it, cost(w)))
            eta /= 2
        it += 1
        g = grad(w)
        w -= eta*g
        if np.linalg.norm(g) < 1e-5:
            break
    return w


w0 = np.random.randn(X0_bar.shape[0], 1)
w_custom = gd(w0, eta=0.01)
print('\nw hinge: ', w_custom.shape)
print('+ w = {}'.format(w_custom[:2, 0]))
print('+ b = {}'.format(w_custom[-1, 0]))

# fit_lib
clf = SVC(kernel='linear', C=C)
clf.fit(X, y)
w_sk = clf.coef_
b_sk = clf.intercept_
print('\nw sklearn:', w_sk.shape)
print('+ w = {}'.format(w_sk[0]))
print('+ b = {}'.format(b_sk[0]))


w0, w1 = w_sk[0]
b = b_sk[0]

# boundaries
boundx = np.array([-1, 6])
boundy = (-w0*boundx - b) / w1
margin1 = (-w0*boundx - b+1) / w1
margin2 = (-w0*boundx - b-1) / w1

# plot
fig, ax = plt.subplots()

X_sup = clf.support_vectors_
for x3, y3 in X_sup:
    circle = plt.Circle((x3, y3), 0.1, color='k', fill=False)
    ax.add_artist(circle)

ax.grid(alpha=0.3)
ax.plot(boundx, boundy, 'k-')
ax.plot(boundx, margin1, 'k:')
ax.plot(boundx, margin2, 'k:')

ax.fill_between(boundx, 8, boundy, color='b', alpha=0.1)
ax.fill_between(boundx, 0, boundy, color='r', alpha=0.1)

ax.plot(X0[:, 0], X0[:, 1], 'bs')
ax.plot(X1[:, 0], X1[:, 1], 'ro')

ax.axis('square')
xlims = np.around(np.array([X[:, 0].min(), X[:, 0].max()]))
ylims = np.around(np.array([X[:, 1].min(), X[:, 1].max()]))
ax.set_ylim(ylims)
ax.set_xlim(xlims)
plt.show()
