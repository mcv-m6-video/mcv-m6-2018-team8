from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import numpy as np

"""
Return Precision, Recall, F1-Score and Accuracy
"""
def metrics(TP, FP, TN, FN, y_gt, y_test):

    precision_list = []
    recall_list = []
    fscore_list = []
    accuracy_list = []

    #confusion_mat = confusion_matrix(y_gt, y_test)
    for m in range(len(y_test)):

        precision = float(TP[m])/float(TP[m]+FP[m])
        recall = float(TP[m])/float(TP[m] + FN[m])
        fscore = 2*(float(precision*recall)/float(precision+recall))
        #precision, recall, fscore, support = precision_recall_fscore_support(y_gt,y_test)
        accuracy = float(TP[m] + TN[m])/float(TP[m] + TN[m] + FP[m] + FN[m])
        #auc = roc_auc_score(y_gt, y_test[m]) # provides a numerical assessement of the quality of the system.
        #area = auc

        precision_list.append(precision)
        recall_list.append(recall)
        fscore_list.append(fscore)
        accuracy_list.append(accuracy)

    return precision_list, recall_list, fscore_list, accuracy_list