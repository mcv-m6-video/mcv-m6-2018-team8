import sys
sys.path.append('../.')
import os
from common.config import *
from common.Database import *
from methods.GaussianMethods import *
from common.extractPerformance import *
from common.metrics import *
from Holefilling import *
import matplotlib.pyplot as plt

def checkFilesNPY(dir="."):
    if not os.path.isfile(os.path.join(dir,"gt_test_{}.npy".format(DATABASE))):
        return False
    if not os.path.isfile(os.path.join(dir,"matrix_mean_{}.npy".format(DATABASE))):
        return False
    if not os.path.isfile(os.path.join(dir,"matrix_std_{}.npy".format(DATABASE))):
        return False

    mean = os.path.join(dir,"matrix_mean_{}.npy".format(DATABASE))
    std = os.path.join(dir, "matrix_std_{}.npy".format(DATABASE))
    gt_test = os.path.join(dir,"gt_test_{}.npy".format(DATABASE))

    return np.load(mean), np.load(std), np.load(gt_test)


if __name__ == "__main__":
    gt_db = Database(abs_dir_gt, start_frame=start_frame, end_frame=end_frame)
    input_db = Database(abs_dir_input, start_frame=start_frame, end_frame=end_frame)
    # results_db = Database(abs_dir_result, start_frame=0)

    gt = gt_db.loadDB(im_color=False)
    input = input_db.loadDB(im_color=False)

    array_rho = np.array([0.22])  # 0.11 for traffic
    array_alpha = np.linspace(0, 5, 10, endpoint=True)
    #    array_rho = np.linspace(0, 1, 10, endpoint=True)

    gt_test = []
    if GAUSSIAN_METHOD in 'adaptative':
        matrix_mean, matrix_std, gt_test = OneSingleGaussianAdapt(input, alpha_params=array_alpha,
                                                                  rho_params=array_rho,
                                                                  im_show=False) if not checkFilesNPY("numpy_files") else checkFilesNPY("numpy_files")
    else:
        matrix_mean, matrix_std, gt_test = OneSingleGaussian(input, array_alpha, im_show=False)

    gt2 = gt[(len(gt) // 2):]

    gt_hole = Holefilling(gt_test, 4)

    TP_list, FP_list, TN_list, FN_list = extractPerformance_2Params(gt2, gt_hole, array_params_a=array_alpha,
                                                                    array_params_b=array_rho,
                                                                    im_show_performance=False)

    precision_list, recall_list, fscore_list, accuracy_list = metrics_2Params(TP_list, FP_list, TN_list, FN_list,
                                                                              array_params_a=array_alpha,
                                                                              array_params_b=array_rho)