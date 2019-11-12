import sys
import numpy as np
import operator


def softmax(s):
    e = np.exp(s - s.max())
    z = e / e.sum()
    return z


class RNN:

    def __init__(self, wdim, hdim, bptt_trunc=4):
        self.wdim = wdim
        self.hdim = hdim
        self.bptt_trunc = bptt_trunc

        self.U = np.random.uniform(-np.sqrt(1./wdim), np.sqrt(1./wdim), (hdim, wdim))
        self.V = np.random.uniform(-np.sqrt(1./hdim), np.sqrt(1./hdim), (wdim, hdim))
        self.W = np.random.uniform(-np.sqrt(1./hdim), np.sqrt(1./hdim), (hdim, hdim))


    def forward(self, x):
        T = len(x)
        s = np.zeros((T+1, self.hdim))
        o = np.zeros((T, self.wdim))

        for t in range(T):
            s[t] = np.tanh(self.U[:, x[t]] + self.W.dot(s[t-1]))
            o[t] = softmax(self.V.dot(s[t]))
        return [o, s]


    def predict(self, x):
        o, s = self.forward(x)
        return o.argmax(axis=1)


    def loss(self, X, Y):
        L = 0
        N = np.sum(len(y) for y in Y)
        for x,y in zip(X, Y):
            o, s = self.forward(x)
            q = o[range(len(x)), y]
            L += -np.sum(np.log(q))
        return L / N


    def bptt(self, x, y):
        T = len(y)
        o, s = self.forward(x)
        dU = np.zeros(self.U.shape)
        dV = np.zeros(self.V.shape)
        dW = np.zeros(self.W.shape)

        dz2 = o
        dz2[np.arange(len(y)), y] -= 1.
        for t in np.arange(T)[::-1]:
            dV += np.outer(dz2[t], s[t])
            dz1 = self.V.T.dot(dz2[t]) * (1 - s[t]**2)

            for j in np.arange(max(0, t-self.bptt_trunc), t+1)[::-1]:
                dW += np.outer(dz1, s[j-1])
                dU[:,x[j]] += dz1
                dz1 = self.W.T.dot(dz1) * (1 - s[j-1]**2)
        return dU, dV, dW


    def gradient_check(self, x, y, eps=0.001, err=0.01):
        g_bptts = self.bptt(x, y)

        pnames = ['U', 'V', 'W']
        for i, pname in enumerate(pnames):
            param = operator.attrgetter(pname)(self)
            g_est = np.zeros_like(param)

            it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index
                p_origin = param[ix]

                param[ix] = p_origin + eps
                l_next = self.loss([x], [y])
                param[ix] = p_origin - eps
                l_prev = self.loss([x], [y])

                g_est[ix] = (l_next - l_prev)/(2*eps)
                # Reset param to original value
                param[ix] = p_origin
                it.iternext()

            g_bptt = g_bptts[i]
            print('diff of {}: {}'.format(pname, (g_bptt - g_est).mean()))


    def update(self, x, y, lr):
        dU, dV, dW = self.bptt(x, y)
        self.U -= lr * dU
        self.V -= lr * dV
        self.W -= lr * dW


def train(model, X_train, Y_train, lr=0.005, epochs=10, verbose=2):
    losses = []
    for epoch in range(epochs):
        for i, (x, y) in enumerate(zip(X_train, Y_train)):
            model.update(x, y, lr)

        if (epoch % verbose == 0):
            loss = model.loss(X_train, Y_train)
            print ("Epoch {}/{} Loss= {}".format(epoch, epochs, loss))
            # Adjust the learning rate if loss increases
            if (len(losses) > 0 and loss > losses[-1]):
                lr = lr * 0.5
                print ("Decay lr to {}".format(lr))
            losses.append(loss)
            sys.stdout.flush()
    return losses
