import numpy as np
import sys
# from scipy.stats import threshold

def performanceW2(gt, gt_test):

    # gt1d = gt.ravel()
    # y_gt = np.zeros((len(gt1d)))

    # for j in range(len(gt1d)):
    #     if gt1d[j] == 0 or gt1d[j] == 50:
    #         y_gt[j] = 0
    #     elif gt1d[j] == 255:
    #         y_gt[j] = 1
    #     elif gt1d[j] == 85 or gt1d[j] == 170:
    #         y_gt[j] = np.nan

    gt = np.where((gt <= 50), 0, gt)
    gt = np.where((gt == 255), 1, gt)
    gt = np.where((gt == 85), -1, gt)
    gt = np.where((gt == 170), -1, gt)

    TP_list = []
    FP_list = []
    TN_list = []
    FN_list = []

    for a in range(len(gt_test)):
        np.sum(np.logical_and(gt_test[a] == 1, gt == 1))
        # True Positive (TP): # pixels correctly segmented as foreground
        TP = np.sum(np.logical_and(gt_test[a] == 1, gt == 1))

        # True Negative (TN): pixels correctly detected as background
        TN = np.sum(np.logical_and(gt_test[a] == 0, gt == 0))

        # False Positive (FP): pixels falsely segmented as foreground
        FP = np.sum(np.logical_and(gt_test[a] == 1, gt == 0))

        # False Negative (FN): pixels falsely detected as background
        FN = np.sum(np.logical_and(gt_test[a] == 0, gt == 1))

        TP_list.append(TP)
        FP_list.append(FP)
        TN_list.append(TN)
        FN_list.append(FN)

        sys.stdout.write("\r>  Computing ... {:.2f}%".format(a*len(gt_test)/100))
        sys.stdout.flush()

    # gt1d = np.ravel(gt)
    # y_gt = np.zeros((len(gt1d)))

    # y_test = np.zeros((len(gt_test), len(y_gt)))

    # for w in range(len(gt_test)):
    #
    #     test1d = gt_test[w].ravel()
    #     y_test[w] = test1d
    #
    #     TP = 0
    #     FP = 0
    #     TN = 0
    #     FN = 0
    #
    #     for i in range(len(y_gt)):
    #         if np.isnan(y_gt[i]) == False:
    #             if y_test[w][i] == 1 and y_gt[i] == 1:
    #                 TP += 1  # pixels correctly segmented as foreground
    #             elif y_test[w][i] == 1 and y_gt[i] == 0:
    #                 FP += 1  # pixels falsely segmented as foreground
    #             elif y_test[w][i] == 0 and y_gt[i] == 0:
    #                 TN += 1  # pixels correctly detected as background
    #             elif y_test[w][i] == 0 and y_gt[i] == 1:
    #                 FN += 1  # pixels falsely detected as background
    #         #elif y_test[i] == 1 and y_gt[i] == "nan":
    #             #K1 += 1
    #         #elif y_test[i] == 0 and y_gt[i] == "nan":
    #             #K2 += 1
    #
    #         if not(i % 100):
    #             sys.stdout.write("\r>  Computing  {:.2f}%".format(100*i/len(y_gt)))
    #             sys.stdout.flush()
    #
    #     TP_list.append(TP)
    #     FP_list.append(FP)
    #     TN_list.append(TN)
    #     FN_list.append(FN)

    # return TP_list, FP_list, TN_list, FN_list, y_gt, y_test

    return TP_list, FP_list, TN_list, FN_list