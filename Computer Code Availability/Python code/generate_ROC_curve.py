
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import plot_confusion_matrix

import config


class_roc = [0,2,3,5,8,9,11,12,13] #adjust according to cod_lit and group

#file address - adjust
logs = pd.read_csv(str(" ",sep=",")


y_real   = logs.cod_lit
y_test_0 = logs.cod_lit_test
y_real = preprocessing.label_binarize(y_real, classes=class_roc) 
y_test_0 = preprocessing.label_binarize(y_test_0, classes=class_roc)
n_classes = y_real.shape[1]

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = metrics.roc_curve(y_real[:, i], y_test_0[:, i])
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])

for iss in range(0, len(class_roc)):
    plt.plot(fpr[iss], tpr[iss],
                 label='ROC curve of Lithology Code {0}'
                 ''.format(class_roc[iss], roc_auc[iss]))                


plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.legend(loc="lower right")

plt.savefig(config.results_graphics + "/curveROC.png")