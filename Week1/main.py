from Database import *
from config import *
from performance import *
from metrics import *
from temporalanalysis import *
from frameperf import *
from FramePerformancedelay import *
# def getLabels(self, images):
    # for im in images:


if __name__ == "__main__":

    start_frame = 1201
    end_frame = 1400
    gt_db = Database(abs_dir_gt, start_frame=start_frame, end_frame=end_frame)
    input_db = Database(abs_dir_input, start_frame=start_frame, end_frame=end_frame)
    results_db = Database(abs_dir_result, start_frame=0)

    gt = gt_db.loadDB(im_color=False)
    input = input_db.loadDB(im_color=False)
    res = results_db.loadDB(im_color=False)

    res_A = res[:200]
    res_B = res[200:]

    print(len(gt))
    print(len(input))
    print(len(res))

#    test = np.abs(gt - res_A)
#    for im in test:
#        cv2.imshow("Test A", im)
#        cv2.waitKey(1)

#    test = np.abs(gt - res_B)
#    for im in test:
#        cv2.imshow("Test B", im)
#        cv2.waitKey(1)

    TP, FP, TN, FN, y_gt, y_test = performance(gt, res_A)

    confusion_mat, precision, recall, fscore, accuracy, auc = metrics(TP, FP, TN, FN, y_gt, y_test)

    TP_list_A, total_foreground_list_A, fscore_list = FramePerformance(gt, res_B)
    #TP_list_B, total_foreground_list_B = FramePerformance(gt, res_B)

    temporalAnalysis(TP_list_A, total_foreground_list_A, fscore_list)

    array_of_delays = [0,5,10,20,30]
    TP_listlist, total_foreground_listlist, fscore_listlist = FramePerformancedelay(gt, res_B,array_of_delays)
    temporalAnalysisDelay(TP_listlist, total_foreground_listlist, fscore_listlist)
