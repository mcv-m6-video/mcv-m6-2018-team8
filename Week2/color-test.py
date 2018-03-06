import sys
sys.path.append('../.')
from common.config import *
from common.Database import *
from OneSingleGaussian import *
from performanceW2 import *
from GaussianColor import *
from metrics import *

if __name__ == "__main__":

    start_frame = 1050
    end_frame = 1350
    gt_db = Database(abs_dir_gt, start_frame=start_frame, end_frame=end_frame)
    input_db = Database(abs_dir_input, start_frame=start_frame, end_frame=end_frame)
    results_db = Database(abs_dir_result, start_frame=0)

    gt = gt_db.loadDB(im_color=False)
    input = input_db.loadDB(im_color=True, color_space="HSV", im_show=True)

    params = np.arange(0, 10, 0.2)
    # matrix_mean, matrix_std, gt_test = GaussianColorRGB(input, array_alpha, im_show=False)
    matrix_mean, matrix_std, gt_test = GaussianColorHSV(input, params)

    gt2 = gt[(len(gt)//2):]
    TP_list, FP_list, TN_list, FN_list = performanceW2(gt2, gt_test, array_params=params)
    precision_list, recall_list, fscore_list, accuracy_list = metrics(TP_list, FP_list, TN_list, FN_list, gt_test, array_params=params)

    plotF1Score(np.linspace(0, params[-1], len(params)), fscore_list)
    plotPrecisionRecall(recall_list, precision_list, label=DATABASE)