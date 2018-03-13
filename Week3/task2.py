import sys
sys.path.append('../.')
from common.config import *
from common.Database import *
from methods.GaussianMethods import *
from common.extractPerformance import *
from common.metrics import *
import matplotlib.pyplot as plt
from AreaFiltering import *

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
    results_db = Database(abs_dir_result, start_frame=0)

    gt = gt_db.loadDB(im_color=False)
    input = input_db.loadDB(im_color=False)

    array_alpha = np.linspace(0.1, 5, 10, endpoint=True)
    array_rho = np.array([0.22])  # Highway
    # array_rho = np.array([0.22]) #Fall
    # array_rho = np.array([0.11]) # Traffic

    gt_train = gt[(len(gt) // 2):]

    gt_test = []
    if GAUSSIAN_METHOD in 'adaptative':
        matrix_mean, matrix_std, gt_test = OneSingleGaussianAdapt(input, alpha_params=array_alpha,
                                                                  rho_params=array_rho, im_show=False) if not checkFilesNPY("numpy_files") else checkFilesNPY("numpy_files")
    else:
        matrix_mean, matrix_std, gt_test = OneSingleGaussian(input, array_alpha, im_show=False)

    area_array = np.linspace(20, 220, 10).astype(np.int)

    gt_filtered = []
    for i, area in enumerate(area_array):

        total_id = 0
        for id_a, alpha in enumerate(array_alpha):
            for id_r, rho in enumerate(array_rho):
                gt_filtered.append(AreaFiltering(gt_test[total_id], area))
                total_id += 1

    TP_list, FP_list, TN_list, FN_list = extractPerformance_2Params(gt_train, gt_filtered,
                                                                    array_params_a=area_array, array_params_b=array_alpha,
                                                                    im_show_performance=False)

    precision_all, recall_all, fscore_list, accuracy_all = metrics_2Params(TP_list, FP_list, TN_list, FN_list,
                                                                           array_params_a=area_array,
                                                                           array_params_b=array_alpha)

    auc_list = getAUC(recall_all, precision_all)

    plotF1Score3D(x_axis=area_array, y_axis=array_alpha, z_axis=fscore_list, x_label='Area', y_label='Alpha', z_label='F1-Score')

    # max Precision-Recall
    plotPrecisionRecall(recall_all[np.argmax(fscore_list) // len(array_alpha)], precision_all[np.argmax(fscore_list) // len(array_alpha)])

    final_images = np.array(gt_filtered)
    MakeYourGIF(final_images[np.argmax(fscore_list)],
                path_to_save='best_case_{}_{}_{}-{}.gif'.format(GAUSSIAN_METHOD, MORPH_EX, MORPH_STRUCTURE, DATABASE))

    plot2D(x_axis=area_array, y_axis=auc_list, x_label="Pixel area", y_label="AUC-PR")