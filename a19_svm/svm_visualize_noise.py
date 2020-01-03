import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
np.random.seed(21)

# data
means = [[2, 2], [4, 1]]
cov = [[.3, .2], [.2, .3]]
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X1[2] = [2.7, 2]
X = np.around(np.vstack([X0, X1]), 3)
y = np.array([1]*N + [-1]*N)

# model
clf = SVC(kernel='linear', C=1e5)
clf.fit(X, y)

w = clf.coef_
b = clf.intercept_

print('w:', w)
print('b:', b)

w0, w1 = w[0]

# boundaries
boundx = np.array([0, 5])
boundy = (-w0*boundx - b) / w1
margin1 = (-w0*boundx - b+1) / w1
margin2 = (-w0*boundx - b-1) / w1

def perpen(x1, y1, x2, y2, x3, y3):
    k = ((x3-x1)*(x2-x1) + (y3-y1)*(y2-y1)) / ((y2-y1)**2 + (x2-x1)**2)
    x4 = x1 + k*(x2-x1)
    y4 = y1 + k*(y2-y1)
    return (x4, y4)

# plot
fig, ax = plt.subplots()

bx1, bx2 = boundx
by1, by2 = boundy
X_sup = clf.support_vectors_
for x3, y3 in X_sup:
    print('support: ', x3, y3)
    x4, y4 = perpen(bx1, by1, bx2, by2, x3, y3)
    ax.plot([x3, x4], [y3, y4], 'k:')
    circle = plt.Circle((x3, y3), 0.1, color='k', fill=False, linestyle=':')
    ax.add_artist(circle)

ax.grid(alpha=.3)
ax.plot(boundx, boundy, 'k-', linewidth=2)
ax.plot(boundx, margin1, 'k--')
ax.plot(boundx, margin2, 'k--')

# better boundary
def parallel(x1, y1, x2, y2, x3, y3):
    # (x1,y4)--(x2,y5) // (x1,y1)--(x2,y2) | include (x3,y3)
    k = (y2 - y1)/(x2 - x1)
    c = y3 - k*x3
    y4 = k*x1 + c
    y5 = k*x2 + c
    return (y4, y5)

x3, y3 = X[15]
circle = plt.Circle((x3, y3), 0.1, color='c', fill=False, linestyle=':')
ax.add_artist(circle)

y4, y5 = parallel(bx1, by1, bx2, by2, x3, y3)
ax.plot([bx1, bx2], [y4, y5], 'c:')

my1, my2 = margin1
y6, y7 = (my1+y4)/2, (my2+y5)/2
ax.plot([bx1, bx2], [y6, y7], 'c')


ax.annotate(s='noise', fontsize=15, xy=(2.8, 2), xytext=(4, 2.5),
             arrowprops=dict(arrowstyle='->'))

ax.fill_between(boundx, 3, boundy, color='b', alpha=.1)
ax.fill_between(boundx, 0, boundy, color='r', alpha=.1)

ax.plot(X0[:, 0], X0[:, 1], 'bs')
ax.plot(X1[:, 0], X1[:, 1], 'ro')

ax.axis('equal')
ax.set_ylim(0, 3)
ax.set_xlim(1, 5)
plt.show()
