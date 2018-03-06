from sklearn.metrics import confusion_matrix, auc
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import numpy as np

"""
Return Precision, Recall, F1-Score and Accuracy
"""
def metrics(TP, FP, TN, FN, y_test, array_params=None):

    precision_list = []
    recall_list = []
    fscore_list = []
    accuracy_list = []

    dim = 1
    if np.array(y_test).ndim > 3:
        dim = len(y_test)

    #confusion_mat = confusion_matrix(y_gt, y_test)
    for m in range(dim):

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

    print("\nSummary: Precision, Recall, F1-Score, Accuracy for each alpha")
    print("------------------------------------------------------------")
    if array_params is None:
        print("Precision: {:.4f} \tRecall: {:.4f} \tF1-Score: {:.4f} \tAccuracy: {:.4f}".format(precision_list[0],recall_list[0],fscore_list[0],accuracy_list[0]))
    else:
        for id, alpha in enumerate(array_params):
            print("For alpha = {} - Precision: {:.4f} \tRecall: {:.4f} \tF1-Score: {:.4f} \tAccuracy: {:.4f}".format(alpha, precision_list[id], recall_list[id], fscore_list[id], accuracy_list[id]))
        print("AUC: {}".format(getAUC(precision_list, recall_list)))
        print("Best F1-Score is {:.4f} with alpha {:.4f}".format(np.max(fscore_list),array_params[np.argmax(fscore_list)]))
        print("------------------------------------------------------------")

    return precision_list, recall_list, fscore_list, accuracy_list


"""
Return the AUC (Area Under Curve)
"""
def getAUC(a,b):
    return auc(a, b)

def plotF1Score(x_axis, y_axis):
    plt.figure()
    plt.plot(x_axis, y_axis, 'b', label='F1-Score')
    plt.legend(loc="lower right")
    plt.xlabel("Alpha")
    plt.ylabel("F1-score")
    plt.axis([0, max(x_axis), 0, max(y_axis)]) # [xmin, xmax, ymin, ymax]
    plt.show()