import numpy as np

def get_entropy(labels):
    vals = np.unique(labels)
    if len(vals) == 1:
        return 0
    else:
        freq = np.array([labels.count(v)/len(labels) for v in vals])
        freq = -np.sum(freq * np.log(freq))
        return freq


class Node(object):
    def __init__(self, name=None, ids=None, parent=None, prop=None, depth=0, childs=None, label=None):
        self.parent = parent
        self.name = name
        self.ids = ids
        self.prop = prop
        self.depth = depth
        self.childs = childs
        self.label = label


class DecisionTree(object):
    def __init__(self, depth_max=2):
        self.depth_max = depth_max

    def fit(self, data, tags):
        N, M = data.shape
        self.tags = tags
        self.data = data
        self.M = M-1
        self.props = np.arange(self.M)
        print('initialized prop + label = {}'.format(self.tags))

        self.root = Node(parent='seed' ,name='root', ids=np.arange(N), depth=0)
        queue = [self.root]
        while queue:
            node = queue.pop(0)
            print('\n\n----- {}: from {}={}, {} -----'.format(node.depth, node.parent, node.name, node.ids))
            if len(np.unique(self.data[node.ids, -1])) == 1:
                node.label = data[node.ids[0], -1]
                print('non split -> all label: {}'.format(node.label))
                continue
            if node.depth >= self.depth_max:
                node.label = self.get_label(data[node.ids, -1].tolist())
                print('non split -> most label: {}'.format(node.label))
                continue
            self.split(node)
            queue += node.childs.values()

    def split(self, node):
        ids = node.ids
        entropies = np.ones(self.M)
        for p in self.props:
            if self.tags[p] == node.parent:
                continue
            vals = np.unique(self.data[ids,p])
            labelss = [self.data[ids[self.data[ids,p]==v],-1] for v in vals]
            entropy = np.sum([len(y)*get_entropy(y.tolist()) for y in labelss])/len(ids)
            entropies[p] = entropy

        print('\t\tentropies', np.around(entropies,2))
        p_best = np.argmin(entropies)
        node.prop = p_best
        vals = np.unique(self.data[ids,p_best])
        print('> p best: ', self.tags[p_best])
        print('> v best: ', vals)
        idss = [ids[self.data[ids,p_best]==v] for v in vals]
        node.childs = {v: Node(parent=self.tags[p_best], name=v, ids=x, depth=node.depth+1) for x,v in zip(idss, vals)}

    def get_label(self, labels):
        vals = np.unique(labels)
        freq = np.array([labels.count(v) for v in vals])
        v_most = vals[np.argmax(freq)]
        return v_most

    def predict(self, X):
        N = X.shape[0]
        labels = []
        for i, x in enumerate(X):
            node = self.root
            while node.childs:
                node = node.childs[x[node.prop]]
            labels.append(node.label)
        return labels


raw_data = np.loadtxt('weather.csv', dtype=str, delimiter=',', skiprows=0)[:,1:]
tags = raw_data[0]
data = raw_data[1:]

tree = DecisionTree(depth_max=5)
print('\n\n----- TRAINING -----\n')
tree.fit(data, tags)

print('\n\n----- TESTING -----\n')
ytrue = data[:,-1].tolist()
ypred = tree.predict(data[:,:-1])
print('pred: {}'.format(ypred))
print('true: {}'.format(ytrue))

