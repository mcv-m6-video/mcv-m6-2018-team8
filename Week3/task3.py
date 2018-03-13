import sys
sys.path.append('../.')
from common.config import *
from common.Database import *
from common.extractPerformance import *
from common.metrics import *
from methods.GaussianMethods import *
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
    results_db = Database(abs_dir_result, start_frame=0)

    gt = gt_db.loadDB(im_color=False)
    input = input_db.loadDB(im_color=False)

    array_alpha = np.linspace(0.1, 5, 10, endpoint=True)
    array_rho = np.array([0.22]) # Highway
    # array_rho = np.array([0.22]) #Fall
    # array_rho = np.array([0.11]) # Traffic

    gt_train = gt[(len(gt) // 2):]

    gt_test = []
    if GAUSSIAN_METHOD in 'adaptative':
        matrix_mean, matrix_std, gt_test = OneSingleGaussianAdapt(input, alpha_params=array_alpha, rho_params=array_rho, im_show=False) if not checkFilesNPY("numpy_files") else checkFilesNPY("numpy_files")
    else:
        matrix_mean, matrix_std, gt_test = OneSingleGaussian(input, array_alpha, im_show=False)


    """
    Sweep the kernel size of the Structure Element for the Morphological Operations
    Type of Structures (change on config.py):
    (examples)
    - cv2.MORPH_RECT (Rectangular)
    - cv2.MORPH_ELLIPSE (Ellipse)
    - cv2.MORPH_CROSS (Cross)
    """
    gt_test_morph = []
    kernels = np.linspace(1, 10, 10).astype(np.uint8)
    for id_k, k in enumerate(kernels):
        kernel = cv2.getStructuringElement(MORPH_STRUCTURE, (k, k))

        total_id = 0
        for id_a, alpha in enumerate(array_alpha):
            for id_r, rho in enumerate(array_rho):
                gt_test_morph.append(MorphologicalTransformation(gt_test[total_id], kernel=kernel, type=MORPH_EX))
                # compareImages(gt_test[total_id], gt_test_morph[total_id], delay_ms=10)
                total_id += 1

    np.save("gt_test_{}_{}-{}".format(GAUSSIAN_METHOD, MORPH_EX, DATABASE), gt_test_morph)

    TP_list, FP_list, TN_list, FN_list = extractPerformance_2Params(gt_train, gt_test_morph,
                                                                    array_params_a=kernels, array_params_b=array_alpha,
                                                                    im_show_performance=False)
    performance_dict = {}
    performance_dict["TP"] = TP_list
    performance_dict["FP"] = FP_list
    performance_dict["TN"] = TN_list
    performance_dict["FN"] = FN_list
    np.save("performanace_{}_{}_{}-{}".format(GAUSSIAN_METHOD, MORPH_EX, MORPH_STRUCTURE, DATABASE), performance_dict)

    precision_list, recall_list, fscore_list, accuracy_list = metrics_2Params(TP_list, FP_list, TN_list, FN_list,
                                                                              array_params_a=kernels, array_params_b=array_alpha)

    metrics_dict = {}
    metrics_dict["Precision"] = precision_list
    metrics_dict["Recall"] = recall_list
    metrics_dict["F1"] = fscore_list
    metrics_dict["Accuracy"] = accuracy_list
    np.save("metric_{}_{}_{}-{}".format(GAUSSIAN_METHOD, MORPH_EX, MORPH_STRUCTURE, DATABASE), metrics_dict)

    parameters = {}
    parameters["Database"] = DATABASE
    parameters["GaussianMethod"] = GAUSSIAN_METHOD
    parameters["Alpha"] = array_alpha
    parameters["Rho"] = array_rho
    parameters["Kernels"] = kernels
    parameters["MorphEx"] = MORPH_EX
    np.save("parameters_{}_{}_{}-{}".format(GAUSSIAN_METHOD, MORPH_EX, MORPH_STRUCTURE, DATABASE), parameters)


    plot_name = "f1score3d_{}_{}_{}-{}.png".format(datetime.now().strftime('%d%m%y_%H-%M-%S'), MORPH_EX, MORPH_STRUCTURE, DATABASE)
    plotF1Score3D(x_axis=kernels, y_axis=array_alpha, z_axis=fscore_list, x_label='Kernel', y_label='Alpha', z_label='F1-score', name=plot_name)
    plotPrecisionRecall(recall_list, precision_list)

    final_images = np.array(gt_test_morph)
    # np.load("metric_{}_{}".format(GAUSSIAN_METHOD, MORPH_EX))
    # np.load("parameters_{}_{}".format(GAUSSIAN_METHOD, MORPH_EX))
    MakeYourGIF(final_images[np.argmax(fscore_list)], path_to_save='best_case_{}_{}_{}-{}.gif'.format(GAUSSIAN_METHOD, MORPH_EX, MORPH_STRUCTURE, DATABASE))