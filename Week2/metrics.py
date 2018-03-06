from sklearn.metrics import confusion_matrix, roc_auc_score, auc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
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
        print("AUC: {}".format(getAUC(recall_list, precision_list, reorder=True)))
        print("Best F1-Score is {:.4f} with alpha {:.4f}".format(np.max(fscore_list),array_params[np.argmax(fscore_list)]))
        print("------------------------------------------------------------")

    return precision_list, recall_list, fscore_list, accuracy_list

def metrics_2Params(TP, FP, TN, FN, array_params_a, array_params_b):

    precision_list = []
    recall_list = []
    fscore_list = []
    accuracy_list = []

    total_id = 0
    for a in range(len(array_params_a)):
        for b in range(len(array_params_b)):

            precision = float(TP[total_id])/float(TP[total_id]+FP[total_id])
            recall = float(TP[total_id])/float(TP[total_id] + FN[total_id])
            fscore = 2*(float(precision*recall)/float(precision+recall))
            accuracy = float(TP[total_id] + TN[total_id])/float(TP[total_id] + TN[total_id] + FP[total_id] + FN[total_id])

            precision_list.append(precision)
            recall_list.append(recall)
            fscore_list.append(fscore)
            accuracy_list.append(accuracy)

            total_id += 1

    print("\nSummary: Precision, Recall, F1-Score, Accuracy for each alpha")
    print("------------------------------------------------------------")
    total_id = 0
    for id_a, param_a in enumerate(array_params_a):
        for id_b, param_b in enumerate(array_params_b):
            print("For params = {:.4f} | {:.4f} - Precision: {:.4f} \tRecall: {:.4f} \tF1-Score: {:.4f} \tAccuracy: {:.4f}".format(param_a, param_b, precision_list[total_id], recall_list[total_id], fscore_list[total_id], accuracy_list[total_id]))
            total_id += 1
    print("AUC: {}".format(getAUC(recall_list, precision_list, reorder=True)))
    print("Best F1-Score is {:.4f} with params = {:.4f} | {:.4f}".format(np.max(fscore_list),array_params_a[np.argmax(fscore_list)%len(array_params_a)], array_params_b[np.argmax(fscore_list)%len(array_params_b)]))
    print("------------------------------------------------------------")

    return precision_list, recall_list, fscore_list, accuracy_list

"""
Return the AUC (Area Under Curve)
"""
def getAUC(a,b, reorder=False):
    return auc(a, b, reorder=reorder)

def plotF1Score2D(x_axis, y_axis):
    plt.figure()
    plt.plot(x_axis, y_axis, 'b', label='F1-Score')
    plt.legend(loc="lower right")
    plt.xlabel("Alpha")
    plt.ylabel("F1-score")
    plt.axis([0, max(x_axis), 0, max(y_axis)]) # [xmin, xmax, ymin, ymax]
    plt.show()

def plotF1Score3D(x_axis, y_axis, z_axis, label=''):

    if isinstance(x_axis, (list,)):
        x_axis = np.array(x_axis)

    if isinstance(y_axis, (list,)):
        y_axis = np.array(y_axis)

    if isinstance(z_axis, (list,)):
        z_axis = np.array(z_axis)

    X, Y = np.meshgrid(x_axis, y_axis)
    Z = z_axis.reshape(X.shape)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    # ax.contour3D(X, Y, Z, 50, cmap='viridis')
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Rho')
    ax.set_zlabel('F1-Score')
    plt.show()
    plt.savefig('f1score3d_' + label + '.png')

def plotPrecisionRecall(recall_axis, precision_axis, label=''):
    plt.figure()
    plt.plot(recall_axis, precision_axis, 'r', label=label)
    plt.legend(loc='upper right')
    plt.title("Precision-Recall")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([np.min(precision_axis), 1])
    plt.xlim([np.min(recall_axis), 1])
    plt.show()
    plt.savefig('precision_recall_' + label + '_a.png')

    plt.figure()
    plt.step(recall_axis, precision_axis, color='b', alpha=0.2, where='post')
    plt.fill_between(recall_axis, precision_axis, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([np.min(precision_axis), 1])
    plt.xlim([np.min(recall_axis), 1])
    plt.title('Precision-Recall curve: AP={0:0.2f}'.format(np.mean(precision_axis)))
    plt.show()
    plt.savefig('precision_recall_' + label + '_b.png')