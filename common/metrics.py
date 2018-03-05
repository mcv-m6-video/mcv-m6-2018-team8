from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

"""
ChangeDetection Metrics
"""
def metrics(TP, FP, TN, FN, y_gt, y_test):

    confusion_mat = confusion_matrix(y_gt, y_test)

    precision = float(TP)/float(TP+FP)
    recall = float(TP)/float(TP + FN)
    fscore = 2*(float(precision*recall)/float(precision+recall))
    #precision, recall, fscore, support = precision_recall_fscore_support(y_gt,y_test)
    accuracy = float(TP + TN)/float(TP + TN + FP + FN)

    auc = roc_auc_score(y_gt, y_test) # provides a numerical assessement of the quality of the system.

    return confusion_mat, precision, recall, fscore, accuracy, auc



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