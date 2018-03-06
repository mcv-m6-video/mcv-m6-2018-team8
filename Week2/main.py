import sys
sys.path.append('../.')
from common.config import *
from common.Database import *
from OneSingleGaussianAdapt_v2 import *
from common.extractPerformance import *
from metrics import *

if __name__ == "__main__":

    gt_db = Database(abs_dir_gt, start_frame=start_frame, end_frame=end_frame)
    input_db = Database(abs_dir_input, start_frame=start_frame, end_frame=end_frame)
    results_db = Database(abs_dir_result, start_frame=0)

    gt = gt_db.loadDB(im_color=False)
    input = input_db.loadDB(im_color=False)

    array_alpha = np.linspace(0, 5, 10, endpoint=True)
    array_rho = np.linspace(0, 1, 10, endpoint=True)

    # matrix_mean, matrix_std, gt_test = OneSingleGaussian(input, array_alpha, im_show=False)
    matrix_mean, matrix_std, gt_test = OneSingleGaussianAdapt(input, alpha_params=array_alpha, rho_params=array_rho, im_show=False)

    gt2 = gt[(len(gt)//2):]
    TP_list, FP_list, TN_list, FN_list = extractPerformance_2Params(gt2, gt_test, array_params_a=array_alpha, array_params_b=array_rho)
    precision_list, recall_list, fscore_list, accuracy_list = metrics_2Params(TP_list, FP_list, TN_list, FN_list, array_params_a=array_alpha, array_params_b=array_rho)

    plotF1Score3D(x_axis=array_alpha, y_axis=array_rho, z_axis=fscore_list, label=DATABASE)
    plotPrecisionRecall(recall_list, precision_list, label=DATABASE)