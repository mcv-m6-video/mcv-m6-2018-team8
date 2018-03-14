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

    array_alpha = np.linspace(0, 5, 10, endpoint=True)
    array_rho = np.array([0.22]) # Highway
    # array_rho = np.array([0.22]) #Fall
    # array_rho = np.array([0.11]) # Traffic

    gt_train = gt[(len(gt) // 2):]

    gt_test = []
    if GAUSSIAN_METHOD in 'adaptative':
        matrix_mean, matrix_std, gt_test = OneSingleGaussianAdapt(input, alpha_params=array_alpha, rho_params=array_rho, im_show=False) if not checkFilesNPY("numpy_files") else checkFilesNPY("numpy_files")
    else:
        matrix_mean, matrix_std, gt_test = OneSingleGaussian(input, array_alpha, im_show=False)

    TP_list, FP_list, TN_list, FN_list = extractPerformance_2Params(gt_train, gt_test,
                                                                    array_params_a=array_alpha, array_params_b=array_rho,
                                                                    im_show_performance=False)

    pr, re, fscore_list, accuracy_all = metrics_2Params(TP_list, FP_list, TN_list, FN_list,
                                                                           array_params_a=array_alpha,
                                                                           array_params_b=array_rho)


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

    TP_list, FP_list, TN_list, FN_list = extractPerformance_2Params(gt_train, gt_test_morph,
                                                                    array_params_a=kernels, array_params_b=array_alpha,
                                                                    im_show_performance=False)

    precision_all, recall_all, fscore_list, accuracy_all = metrics_2Params(TP_list, FP_list, TN_list, FN_list,
                                                                              array_params_a=kernels, array_params_b=array_alpha)

    # Get AUC-List in a sorted way in 2D (for each parameter)
    auc_list = getAUC(recall_all, precision_all)

    plot_name = "f1score3d_{}_{}_{}-{}.png".format(datetime.now().strftime('%d%m%y_%H-%M-%S'), MORPH_EX, MORPH_STRUCTURE, DATABASE)
    plotF1Score3D(x_axis=kernels, y_axis=array_alpha, z_axis=fscore_list, x_label='Kernel',
                  y_label='Alpha', z_label='F1-score', name=plot_name)

    plotPrecisionRecall(recall_all[np.argmax(fscore_list) // len(array_alpha)], precision_all[np.argmax(fscore_list) // len(array_alpha)])

    final_images = np.array(gt_test_morph)
    MakeYourGIF(final_images[np.argmax(fscore_list)], path_to_save='best_case_{}_{}_{}-{}.gif'.format(GAUSSIAN_METHOD, MORPH_EX, MORPH_STRUCTURE, DATABASE))

    plot2D(x_axis=kernels, y_axis=auc_list, x_label="Kernel size", y_label="AUC-PR")

    # Comparision
    plt.figure()
    plt.plot(recall_all[np.argmax(fscore_list) // len(array_alpha)],
             precision_all[np.argmax(fscore_list) // len(array_alpha)], 'g', label="Baseline + MorphEx (AUC {0:.2f})".format(np.max(auc_list)))
    plt.plot(re, pr, '--b', label="Baseline (AUC {0:.2f})".format(getAUC(re.reshape(10),pr.reshape(10))))
    plt.legend(loc='lower left')
    plt.title("Precision-Recall")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0, 1])
    # plt.xlim([0, 1])
    plt.savefig("baseline_vs_other_{}_{}_a.png".format(DATABASE, datetime.now().strftime('%d%m%y_%H-%M-%S')), bbox_inches='tight', frameon=False)
    plt.show()