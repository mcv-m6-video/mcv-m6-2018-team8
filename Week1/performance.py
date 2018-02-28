import numpy as np
import sys
#from scipy.stats import threshold

def performance(gt, test):

    gt1d = gt.ravel()
    test1d = test.ravel()

    #y_hat = np.array(threshold(gt1d, 170))
    #y_actual = np.array(threshold(test1d, 170))

    y_gt = np.where(gt1d >= 170, 1, 0)
    y_test = np.where(test1d >= 170, 1, 0)

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    # total = len(y_gt)
    for i in range(len(y_gt)):
        if y_test[i]==1 and y_gt[i]==1:
           TP += 1 # pixels correctly segmented as foreground
        elif y_test[i]==1 and y_gt[i]==0:
           FP += 1 # pixels falsely segmented as foreground
        elif y_test[i]==0 and y_gt[i]==0:
           TN += 1 # pixels correctly detected as background
        elif y_test[i]==0 and y_gt[i]==1:
           FN += 1 # pixels falsely detected as background

        # sys.stdout.write("\r>  Computing  {:.2f}%".format(100*i/total, 100))
        # sys.stdout.flush()

    return TP, FP, TN, FN, y_gt, y_test
