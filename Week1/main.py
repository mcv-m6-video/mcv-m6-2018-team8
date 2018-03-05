import sys
sys.path.append('../.')
from common.config import *
from common.Database import *
from common.metrics import *

from performance import *
from temporalanalysis import *
from frameperf import *
from FramePerformancedelay import *

if __name__ == "__main__":

    if DATABASE == "changedetection":
        start_frame = 1201
        end_frame = 1400

        gt_db = Database(abs_dir_gt, start_frame=start_frame, end_frame=end_frame)
        input_db = Database(abs_dir_input, start_frame=start_frame, end_frame=end_frame)
        results_db = Database(abs_dir_result, start_frame=0)

        gt = gt_db.loadDB()
        input = input_db.loadDB()
        res = results_db.loadDB()

        res_A = res[:200]
        res_B = res[200:]

        TP, FP, TN, FN, y_gt, y_test = performance(gt, res_A)

        confusion_mat, precision, recall, fscore, accuracy, auc = metrics(TP, FP, TN, FN, y_gt, y_test)

        TP_list_A, total_foreground_list_A, fscore_list = FramePerformance(gt, res_B)
        # TP_list_B, total_foreground_list_B = FramePerformance(gt, res_B)

        temporalAnalysis(TP_list_A, total_foreground_list_A, fscore_list)

        array_of_delays = [0, 5, 10, 20, 30]
        TP_listlist, total_foreground_listlist, fscore_listlist = FramePerformancedelay(gt, res_B, array_of_delays)
        temporalAnalysisDelay(TP_listlist, total_foreground_listlist, fscore_listlist)

    elif DATABASE == "kitti":
        start_frame = 0
        end_frame = -1

        gt_db = Database(abs_dir_gt, start_frame=start_frame, end_frame=end_frame)
        input_db = Database(abs_dir_input, start_frame=start_frame, end_frame=end_frame)
        results_db = Database(abs_dir_result, start_frame=0)

        gt = gt_db.loadDB(im_color=cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
        input = input_db.loadDB(im_color=True)
        res = results_db.loadDB(im_color=cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)

        gt_0 = gt[0]
        gt_1 = gt[1]
        res_0 = res[0]
        res_1 = res[1]

        output_gt_0, output_pred_0, E_0, pepn_0 = MSEN_PEPN(y_gt=gt_0, y_pred=res_0, show_error=True)
        output_gt_1, output_pred_1, E_1, pepn_1 = MSEN_PEPN(y_gt=gt_1, y_pred=res_1, show_error=True)

        print("Error MSEN Seq45  {}%".format(np.mean(E_0)))
        print("Error MSEN Seq157 {}%".format(np.mean(E_1)))
        print("Error PEPN Seq45  {}%".format(pepn_0 * 100))
        print("Error PEPN Seq157 {}%".format(pepn_1 * 100))

        plotHistogram(E_0, pepn_0, "Seq 45")
        plotHistogram(E_1, pepn_1, "Seq 157")

        # ShowOpticalFlow(res_0)

        cv2.imshow("Seq 45", output_pred_0)
        cv2.imshow("Seq 157", output_pred_1)
        cv2.waitKey(1)