import sys
sys.path.append('../.')
from common.config import *
from common.Database import *
from OneSingleGaussianAdapt import *
from common.extractPerformance import *
from common.metrics import *
from MorphologicTransformation import *

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

    array_alpha = np.linspace(0, 5, 10, endpoint=True)
    array_rho = np.array([0.22]) # Highway
    # array_rho = np.array([0.22]) #Fall
    # array_rho = np.array([0.11]) # Traffic

    gt_train = gt[(len(gt) // 2):]

    gt_test = []
    if GAUSSIAN_METHOD in 'adaptative':
        matrix_mean, matrix_std, gt_test = OneSingleGaussianAdapt(input, alpha_params=array_alpha, rho_params=array_rho, im_show=False) if not checkFilesNPY("numpy_files") else checkFilesNPY("numpy_files")
    # else:
        # matrix_mean, matrix_std, gt_test = OneSingleGaussian(input, array_alpha, im_show=False)

    kernels = np.linspace(1,10,10).astype(np.uint8)
    # cv2.getStructuringElement()

    gt_test_morph = []

    for id_k, k in enumerate(kernels):
        kernel = np.ones((k, k), np.uint8)

        total_id = 0
        for id_a, alpha in enumerate(array_alpha):
            for id_r, rho in enumerate(array_rho):
                gt_test_morph.append(MorphologicalTransformation(gt_test[total_id], kernel=kernel, type=MORPH_EX))
                # compareImages(gt_test[total_id], gt_test_morph[total_id], delay_ms=10)
                total_id += 1

    np.save("gt_test_" + GAUSSIAN_METHOD + "_" + MORPH_EX, gt_test_morph)

    TP_list, FP_list, TN_list, FN_list = extractPerformance_2Params(gt_train, gt_test_morph,
                                                                    array_params_a=kernels, array_params_b=array_alpha,
                                                                    im_show_performance=False)
    performance_dict = {}
    performance_dict.update({"TP":TP_list})
    performance_dict.update({"FP":FP_list})
    performance_dict.update({"TN":TN_list})
    performance_dict.update({"FN":FN_list})
    np.save("performanace_" + GAUSSIAN_METHOD + "_" + MORPH_EX, performance_dict)

    precision_list, recall_list, fscore_list, accuracy_list = metrics_2Params(TP_list, FP_list, TN_list, FN_list,
                                                                              array_params_a=kernels, array_params_b=array_alpha)

    metrics_dict = {}
    metrics_dict.update({"Precision": precision_list})
    metrics_dict.update({"Recall": recall_list})
    metrics_dict.update({"F1": fscore_list})
    metrics_dict.update({"Accuracy": accuracy_list})
    np.save("metric_" + GAUSSIAN_METHOD + "_" + MORPH_EX, metrics_dict)

    parameters = {}
    parameters.update({"Database":DATABASE})
    parameters.update({"GaussianMethod": GAUSSIAN_METHOD})
    parameters.update({"Alpha": array_alpha})
    parameters.update({"Rho": array_rho})
    parameters.update({"Kernels": kernels})
    parameters.update({"MorphEx": MORPH_EX})

    np.save("parameters", parameters)

    plotF1Score3D(x_axis=kernels, y_axis=array_alpha, z_axis=fscore_list, x_label='Kernel', y_label='Alpha', z_label='F1-score')
    plotPrecisionRecall(recall_list, precision_list, label=DATABASE)
