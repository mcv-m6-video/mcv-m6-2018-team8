import sys
sys.path.append('../.')
from common.config import *
from common.Database import *
from common.extractPerformance import *
from common.metrics import *
from methods.GaussianMethods import *
from methods.ShadowRemoval import *
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
    input_color = input_db.loadDB(im_color=True, color_space="BGR")

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

    """ Testing with Guassian Approach from Week2"""
    TP_list, FP_list, TN_list, FN_list = extractPerformance_2Params(gt_train, gt_test,
                                                                    array_params_a=array_alpha,
                                                                    array_params_b=array_rho,
                                                                    im_show_performance=False)

    precision, recall, fscore, _ = metrics_2Params(TP_list, FP_list, TN_list, FN_list,
                                                                      array_params_a=array_alpha,
                                                                      array_params_b=array_rho)

    MakeYourGIF(gt_test[np.argmax(fscore)],
                path_to_save='baseline_{}-{}.gif'.format(GAUSSIAN_METHOD, DATABASE))

    input_color_test = input_color[(len(gt) // 2):]

    gt_test_shadow = ShadowRemoval(input_color_test, gt_test, array_alpha, array_rho, im_show=False)

    TP_list, FP_list, TN_list, FN_list = extractPerformance_2Params(gt_train, gt_test_shadow,
                                                                    array_params_a=array_alpha,
                                                                    array_params_b=array_rho,
                                                                    im_show_performance=False)

    precision_shadow, recall_shadow, fscore_shadow, _ = metrics_2Params(TP_list, FP_list, TN_list, FN_list,
                                                        array_params_a=array_alpha,
                                                        array_params_b=array_rho)

    MakeYourGIF(gt_test_shadow[np.argmax(fscore_shadow)],
                path_to_save='shadow_{}-{}.gif'.format(GAUSSIAN_METHOD, DATABASE))

    """ Testing with Morhphologic Approach from Task 3"""
    gt_test_morph = []
    kernels = np.linspace(1, 10, 10).astype(np.uint8)
    for id_k, k in enumerate(kernels):
        kernel = cv2.getStructuringElement(MORPH_STRUCTURE, (k, k))
        total_id = 0
        for id_a, alpha in enumerate(array_alpha):
            for id_r, rho in enumerate(array_rho):
                gt_test_morph.append(MorphologicalTransformation(gt_test[total_id], kernel=kernel, type=MORPH_EX))
                total_id += 1

    TP_list, FP_list, TN_list, FN_list = extractPerformance_2Params(gt_train, gt_test_morph,
                                                                    array_params_a=kernels, array_params_b=array_alpha,
                                                                    im_show_performance=False)

    precision_morph, recall_morph, fscore_morph, _ = metrics_2Params(TP_list, FP_list, TN_list, FN_list,
                                                                           array_params_a=kernels,
                                                                           array_params_b=array_alpha)

    auc_morph = getAUC(recall_morph, precision_morph)

    MakeYourGIF(gt_test_morph[np.argmax(fscore_morph)],
                path_to_save='morph_{}-{}.gif'.format(GAUSSIAN_METHOD, DATABASE))

    gt_test_morph = np.array(gt_test_morph)
    gt_test_morph_shadow_all = []
    for id_k, k in enumerate(kernels):
        gt_test_morph_shadow = ShadowRemoval(input_color_test, gt_test_morph[id_k:id_k+len(kernels)], array_alpha, array_rho, im_show=False)
        gt_test_morph_shadow_all.append(gt_test_morph_shadow)

    gt_test_morph_shadow_all = np.array(gt_test_morph_shadow_all).reshape(gt_test_morph.shape)

    TP_list, FP_list, TN_list, FN_list = extractPerformance_2Params(gt_train, gt_test_morph_shadow_all,
                                                                    array_params_a=kernels, array_params_b=array_alpha,
                                                                    im_show_performance=False)

    precision_morph_shadow, recall_morph_shadow, fscore_morph_shadow, accuracy_all = metrics_2Params(TP_list, FP_list, TN_list, FN_list,
                                                                           array_params_a=kernels,
                                                                           array_params_b=array_alpha)

    auc_morph_shadow = getAUC(recall_morph_shadow, precision_morph_shadow)

    MakeYourGIF(gt_test_morph_shadow_all[np.argmax(fscore_morph_shadow)],
                path_to_save='morph_shadow_{}-{}.gif'.format(GAUSSIAN_METHOD, DATABASE))

    # Comparision
    plt.figure()
    plt.plot(recall_morph_shadow[np.argmax(fscore_morph_shadow) // len(array_alpha)], precision_morph_shadow[np.argmax(fscore_morph_shadow) // len(array_alpha)], 'g',
             label="Baseline + MorphEx + Shadow Removal (AUC {0:.4f})".format(np.max(auc_morph_shadow)))
    plt.plot(recall_morph[np.argmax(fscore_morph) // len(array_alpha)], precision_morph[np.argmax(fscore_morph) // len(array_alpha)], '--g',
             label="Baseline + MorphEx (AUC {0:.4f})".format(np.max(auc_morph)))
    plt.plot(recall_shadow, precision_shadow, 'b', label="Baseline + Shadow Removal (AUC {0:.4f})".format( getAUC(recall_shadow.reshape(len(array_alpha)), precision_shadow.reshape(len(array_alpha)))))
    plt.plot(recall, precision, '--b', label="Baseline (AUC {0:.4f})".format(getAUC(recall.reshape(len(array_alpha)), precision.reshape(len(array_alpha)))))

    plt.legend(loc='lower left')
    plt.title("Precision-Recall")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0, 1])
    plt.savefig("baselineshadow_vs_other_{}_{}_a.png".format(DATABASE, datetime.now().strftime('%d%m%y_%H-%M-%S')),
                bbox_inches='tight', frameon=False)
    plt.show()