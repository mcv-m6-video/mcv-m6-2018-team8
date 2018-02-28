from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score


def metrics(TP, FP, TN, FN, y_gt, y_test):

    confusion_mat = confusion_matrix(y_gt, y_test)

    precision = float(TP)/float(TP+FP)
    recall = float(TP)/float(TP + FN)
    fscore = 2*(float(precision*recall)/float(precision+recall))
    #precision, recall, fscore, support = precision_recall_fscore_support(y_gt,y_test)
    accuracy = float(TP + TN)/float(TP + TN + FP + FN)

    auc = roc_auc_score(y_gt, y_test) # provides a numerical assessement of the quality of the system.

    return confusion_mat, precision, recall, fscore, accuracy, auc