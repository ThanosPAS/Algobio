import torch
from torch.autograd import Variable
import torch.nn as nn
#import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, matthews_corrcoef

import seaborn as sns

from scipy import stats

# BINDER_THRESHOLD (from Helle - I guess we use this, but not sure what the best argumentation would be)
BINDER_THRESHOLD = 0.426

#fill out
x_test =
y_test =
pred =
targets =

### ROC/AUC
## net.eval depending on method - here ANN
net.eval()
pred = net(x_test)
loss = criterion(pred, y_test)

plot_target_values(data=[(pd.DataFrame(pred.data.numpy(), columns=['target']), 'Prediction'),
                         (test_raw, 'Target')])

y_test_class = np.where(y_test.flatten() >= BINDER_THRESHOLD, 1, 0)
y_pred_class = np.where(pred.flatten() >= BINDER_THRESHOLD, 1, 0)

fpr, tpr, threshold = roc_curve(y_test_class, pred.flatten().detach().numpy())
roc_auc = auc(fpr, tpr)


def plot_roc_curve():
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

plot_roc_curve()


### SCC and PCC

scc = stats.spearman(pred, targets)
def plot_scc():
    plt.title('Spearmans Correlation Coefficient')
    plt.scatter(y_test.flatten().detach().numpy(), pred.flatten().detach().numpy(), label = 'SCC = %0.2f' % scc)
    plt.legend(loc = 'lower right')
    plt.ylabel('Predicted')
    plt.xlabel('Validation targets')
    plt.show()


pcc = stats.pearsonr(pred, targets)
def plot_scc():
    plt.title('Pearsons Correlation Coefficient')
    plt.scatter(y_test.flatten().detach().numpy(), pred.flatten().detach().numpy(), label = 'PCC = %0.2f' % pcc)
    plt.legend(loc = 'lower right')
    plt.ylabel('Predicted')
    plt.xlabel('Validation targets')
    plt.show()


### Spearman
if stats.shapiro(pred)[1] < 0.05 and stats.shapiro(targets)[1] < 0.05:
    print('Preds and targets normally distributed (according to shapiro)')
    print('PCC', stats.pearsonr(pred, targets))
    lot_pcc()
    print('SCC', stats.spearman(pred, targets))
    plot_scc()
else:
    print('Preds and targets not normally distributed (according to shapiro) -> only do Spearman')
    print('SCC', stats.spearman(pred, targets))
    plot_scc()
