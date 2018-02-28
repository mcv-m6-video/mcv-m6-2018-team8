
import numpy as np

def FramePerformance(gt, res):
    TP_list = []
    total_foreground_list = []
    fscore_list = []
    for i in range(len(res)):
        TP = 0
        FP = 0
        FN = 0
        fscore = 0
        gt_th = np.where(gt[i] >= 170, 1, 0)
        res_th = np.where(res[i] >= 170, 1, 0)

        for x in range (res_th.shape[0]):
            for y in range(res_th.shape[1]):
                if res_th[x][y]==1 and gt_th[x][y]==1:
                    TP += 1 # pixels correctly segmented as foreground
                elif res_th[x][y]==1 and gt_th[x][y]==0:
                    FP += 1 # pixels falsely segmented as foreground
                # elif res_th[x][y]==0 and gt_th[x][y]==0:
                #     TN += 1  # pixels correctly detected as background
                elif res_th[x][y]==0 and gt_th[x][y]==1:
                    FN += 1  # pixels falsely detected as background

        if (TP + FP) != 0.0:
            precision = float(TP) / float(TP + FP)
        else:
            precision = 0.0

        if (TP + FN) != 0.0:
            recall = float(TP) / float(TP + FN)
        else:
            recall = 0.0

        if (precision + recall) != 0.0:
            fscore = 2 * (float(precision * recall) / float(precision + recall))
        else:
            fscore = 0.0

        fscore_list.append(fscore)
        TP_list.append(TP)
        total_foreground = TP + FP
        total_foreground_list.append(total_foreground)

    return TP_list, total_foreground_list, fscore_list




