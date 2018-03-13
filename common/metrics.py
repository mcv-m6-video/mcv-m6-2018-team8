from sklearn.metrics import confusion_matrix, roc_auc_score, auc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
from datetime import datetime
from common.config import *
import warnings

"""
Read OpticalFLow uint16 images

flow_u(u,v) = ((float)I(u,v,1)-2^15)/64.0;
flow_v(u,v) = ((float)I(u,v,2)-2^15)/64.0;
valid(u,v)  = (bool)I(u,v,3);
"""
def ReadOpticalFlow(image):
    assert (len(image.shape) == 3)

    image_f64 = np.zeros(image.shape, dtype=np.float)
    image_u16 = image.astype(np.uint16) #unsigned 16bits

    # b g r
    u_comp = image_u16[:,:,2]
    v_comp = image_u16[:,:,1]
    valid_comp = image_u16[:,:,0]

    flow_u = ((u_comp - np.float_power(2,15))/64.0)
    flow_v = ((v_comp - np.float_power(2,15))/64.0)
    valid = (valid_comp == 1) # boolean valid

    # just fot assure only valid pixels
    flow_u[valid_comp == 0] = 0
    flow_v[valid_comp == 0] = 0

    image_f64[:,:,0] = flow_u
    image_f64[:,:,1] = flow_v
    image_f64[:,:,2] = flow_u

    return flow_u, flow_v, valid, image_f64

"""
MeanSquareError and Percentage of Erroneous Pixels in Non-Occluded Areas 
"""
def MSEN_PEPN(y_gt, y_pred, show_error=False, th=3):

    assert(y_gt.shape == y_pred.shape)

    flow_u_gt , flow_v_gt, valid_gt, flow_gt_image = ReadOpticalFlow(y_gt)
    flow_u_pred, flow_v_pred, valid_pred, flow_pred_image = ReadOpticalFlow(y_pred)

    # Errors (mean-square error)
    E_u = np.float_power((flow_u_gt - flow_u_pred), 2)
    E_v = np.float_power((flow_v_gt - flow_v_pred), 2)
    E = np.sqrt(E_u + E_v)
    E[valid_gt==0] = 0
    error_mse = E[valid_gt==1]

    if show_error:
        plt.figure()
        plt.imshow(np.reshape(E, y_gt.shape[:-1]), cmap='viridis')
        plt.colorbar(orientation='horizontal')
        plt.show()

    pepn_error = len(E[E>th]) / len(E[valid_gt])

    return flow_gt_image, flow_pred_image, error_mse, pepn_error

"""
Plot Histogram for the MSE
"""
def plotHistogram(error, pepn_error, name_seq):
    E = error.flatten() if len(error.shape) > 1 else error
    plt.figure()
    cm = plt.cm.get_cmap('viridis')
    n, bins, patches = plt.hist(E, bins=25, normed=True, color='green')
    col = (n - n.min()) / (n.max() - n.min())
    for c, p in zip(np.flip(col,0), patches):
        plt.setp(p, 'facecolor', cm(c))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) # add %
    plt.xlabel("MSEN values")
    plt.ylabel("Number of Pixels")
    plt.title("Histogram {} with a PEPN {:1f}%".format(name_seq, pepn_error * 100))
    plt.show()

"""
Show the OPtical Flow using quiver
"""
def ShowOpticalFlow(image):

    u_flow, v_flow, _, _ = ReadOpticalFlow(image)
    rows, cols = u_flow.shape
    x, y = np.meshgrid(np.arange(0, cols, 1), np.arange(0, rows, 1))
    plt.quiver(x, y, np.abs(u_flow), np.abs(v_flow), scale=1, hatch=' ', alpha=0.3, linewidth=0.001)
    plt.show()

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
        accuracy = float(TP[m] + TN[m])/float(TP[m] + TN[m] + FP[m] + FN[m])

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
    print("AUC: {}".format(getAUC(recall_list, precision_list)))
    print("Best F1-Score is {:.4f} with params = {:.4f} | {:.4f}".format(np.max(fscore_list),array_params_a[np.argmax(fscore_list)//len(array_params_b)], array_params_b[np.argmax(fscore_list)%len(array_params_b)]))
    print("Worst F1-Score is {:.4f} with params = {:.4f} | {:.4f}".format(np.min(fscore_list), array_params_a[
        np.argmin(fscore_list) // len(array_params_b)], array_params_b[np.argmin(fscore_list) % len(array_params_b)]))
    print("------------------------------------------------------------")

    return precision_list, recall_list, fscore_list, accuracy_list

"""
Return the value of the parameter given the index. Useful to depict the desire GIF o plot
"""
def findParams(data, index):
    if index < len(data):
        warnings.warn("index < len(data)")
        return -1
    else:
        return data[index]

"""
Return the AUC (Area Under Curve)
"""
def getAUC(a,b, reorder=False):
    return np.trapz(a, b)
    # return auc(a, b, reorder=reorder)

def plotF1Score2D(x_axis, y_axis):
    plt.figure()
    plt.plot(x_axis, y_axis, 'b', label='F1-Score')
    plt.legend(loc="lower right")
    plt.xlabel("Alpha")
    plt.ylabel("F1-score")
    plt.axis([0, 1, 0, 1]) # [xmin, xmax, ymin, ymax]
    plt.savefig("f1score2d_{}.png".format(datetime.now().strftime('%d%m%y_%H-%M-%S')), bbox_inches='tight', frameon=False)
    # plt.show()

def plotF1Score3D(x_axis, y_axis, z_axis, x_label='', y_label='', z_label='', name=''):

    name = "f1score3d_{}_{}.png".format(datetime.now().strftime('%d%m%y_%H-%M-%S'), DATABASE) if name == '' else name

    if isinstance(x_axis, (list,)):
        x_axis = np.array(x_axis)

    if isinstance(y_axis, (list,)):
        y_axis = np.array(y_axis)

    if isinstance(z_axis, (list,)):
        z_axis = np.array(z_axis)

    X, Y = np.meshgrid(x_axis, y_axis, indexing='ij')
    Z = z_axis.reshape(X.shape)

    plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.contour3D(X, Y, Z, 50, cmap='viridis')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    plt.colorbar(surf)
    plt.savefig(name, bbox_inches='tight', frameon=False)
    plt.show()

def plotPrecisionRecall(recall_axis, precision_axis):
    plt.figure()
    plt.plot(recall_axis, precision_axis, 'r')
    plt.legend(loc='upper right')
    plt.title("Precision-Recall")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([np.min(precision_axis), 1])
    plt.xlim([np.min(recall_axis), 1])
    plt.savefig("precision_recall_{}_a.png".format(DATABASE), bbox_inches='tight', frameon=False)
    # plt.show()

    plt.figure()
    plt.step(recall_axis, precision_axis, color='b', alpha=0.2, where='post')
    plt.fill_between(recall_axis, precision_axis, step='post', alpha=0.2, color='b')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([np.min(precision_axis), 1])
    plt.xlim([np.min(recall_axis), 1])
    plt.title("Precision-Recall curve: AP={0:0.2f}".format(np.mean(precision_axis)))
    plt.savefig("precision_recall_{}_b.png".format(DATABASE), bbox_inches='tight', frameon=False)
    # plt.show()