import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from itertools import cycle

n0, n1 = 20, 30
y_true = np.r_[[0]*n0, [1]*n1]
p_pred = np.r_[np.random.rand(n0)/2, np.random.rand(n1)/2+.2]
fpr, tpr, thresholds = roc_curve(y_true, p_pred, pos_label = 1)


plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)'%auc(fpr, tpr))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()