import sys
sys.path.append('../.')
from common.config import *
from common.Database import *
from common.extractPerformance import *
from common.metrics import *
from methods.GaussianMethods import *
from Week3.MorphologicTransformation import *
from CompensateImage import *
from methods.Farneback import *
from methods.LucasKanade import *
from methods.BlockMatchingOF import *

if __name__ == "__main__":
    gt_db = Database(abs_dir_gt, start_frame=start_frame, end_frame=end_frame)
    input_db = Database(abs_dir_input, start_frame=start_frame, end_frame=end_frame)
    results_db = Database(abs_dir_result, start_frame=0)

    gt = gt_db.loadDB(im_color=False)
    input = input_db.loadDB(im_color=False)

    gt_train = gt[(len(gt) // 2):]

    """
    OpticalFlow Computation (Input images)
    """
    motion_images = []
    for i in range(0, len(gt)-1, 1):

        sys.stdout.write("\r>  {}/{} ... {:.2f}%".format(i, len(gt), i * 100 / len(gt)))
        sys.stdout.flush()

        if OF_TYPE == "LucasKanade":
            y_motion_pred = lucas_kanade_np(input[i], input[i + 1], win=11)
        elif OF_TYPE == "Farneback":
            y_motion_pred = Farneback(input[i], input[i + 1], win=11)
        elif OF_TYPE == "BlockMatching":
            x, y, y_motion_pred = BlockMatchingOpticalFlow(input[i], input[i + 1], window_size=7, stride=1,
                                                           shift=7, im_show_debug=False)
        else:
            raise ValueError("{} does not exist!".format(OF_TYPE))

        motion_images.append(y_motion_pred)

    motion_images = np.array(motion_images)
    np.save("motion_{}".format(OF_TYPE), motion_images)

    """
    Compensation Images (Input images)
    """
    udir_accum = 0
    vdir_accum = 0
    input_comp = []
    for i in range(0, len(input) - 1, 1):
        comp_img, udir_accum, vdir_accum = CompensateImage(input[i], motion_images[..., 0][i],
                                                           motion_images[..., 1], udir_accum=0, vdir_accum=0)

        cv2.imshow("Fixed", comp_img)
        cv2.waitKey(1)
        input_comp.append(comp_img)

    input_comp = np.array(input_comp)

    MakeYourGIF(input_comp, path_to_save='compensate_input_{}.gif'.format(OF_TYPE))
    MakeYourGIF(input, path_to_save='no-compensate_input_{}.gif'.format(OF_TYPE))

    np.save("input_comp_{}.npy".format(OF_TYPE), input_comp)

    """
    OpticalFlow Computation (Ground Truth)
    """
    motion_images_gt = []
    for i in range(0, len(gt)-1, 1):

        sys.stdout.write("\r>  {}/{} ... {:.2f}%".format(i, len(gt), i * 100 / len(gt)))
        sys.stdout.flush()

        if OF_TYPE == "LucasKanade":
            y_motion_pred = lucas_kanade_np(gt[i], gt[i + 1], win=7)
        elif OF_TYPE == "Farneback":
            y_motion_pred = Farneback(gt[i], gt[i + 1], win=7)
        elif OF_TYPE == "BlockMatching":
            x, y, y_motion_pred = BlockMatchingOpticalFlow(gt[i], gt[i + 1], window_size=7, stride=1,
                                                           shift=7, im_show_debug=False)
        else:
            raise ValueError("{} does not exist!".format(OF_TYPE))

        motion_images_gt.append(y_motion_pred)

    motion_images_gt = np.array(motion_images_gt)
    np.save("motion_gt_{}".format(OF_TYPE), motion_images_gt)

    """
    Compensation Images (Ground Truth)
    """
    udir_accum = 0
    vdir_accum = 0
    gt_train_comp = []
    for i in range(0, len(input) - 1, 1):
        comp_img, udir_accum, vdir_accum = CompensateImage(gt[i], motion_images_gt[..., 0][i],
                                                           motion_images_gt[..., 1][i], udir_accum=0, vdir_accum=0)

        cv2.imshow("Fixed", comp_img)
        cv2.waitKey(1)
        gt_train_comp.append(comp_img)

    gt_comp_train = np.array(gt_train_comp)

    final_images = np.array(gt_comp_train)
    MakeYourGIF(final_images, path_to_save='compensate_gt_{}.gif'.format(OF_TYPE))
    MakeYourGIF(gt, path_to_save='no-compensate_gt_{}.gif'.format(OF_TYPE))

    np.save("gt_comp_{}.npy".format(OF_TYPE), gt_train_comp)

    ##########################################################################
    ##########################################################################

    """
    AdaptativeGaussian Computation (no compensation)
    """
    array_alpha = np.linspace(0, 5, 10, endpoint=True)
    # array_rho = np.array([0.22]) # Highway
    # array_rho = np.array([0.22]) #Fall
    array_rho = np.array([0.11]) # Traffic

    if not os.path.exists("gt_test_{}".format(OF_TYPE)):

        gt_test = []
        if GAUSSIAN_METHOD in 'adaptative':
            matrix_mean, matrix_std, gt_test = OneSingleGaussianAdapt(input, alpha_params=array_alpha, rho_params=array_rho, im_show=False)
        else:
            matrix_mean, matrix_std, gt_test = OneSingleGaussian(input, array_alpha, im_show=False)

        np.save("gt_test_{}".format(GAUSSIAN_METHOD), gt_test)

    else:
        gt_test = np.load("gt_test_{}".format(GAUSSIAN_METHOD))

    """
    AdaptativeGaussian Computation (compensation)
    """
    array_alpha = np.linspace(0, 5, 10, endpoint=True)
    # array_rho = np.array([0.22]) # Highway
    # array_rho = np.array([0.22]) #Fall
    array_rho = np.array([0.11]) # Traffic

    if not os.path.exists("gt_test_comp_{}".format(OF_TYPE)):

        if GAUSSIAN_METHOD in 'adaptative':
            matrix_mean, matrix_std, gt_test_comp = OneSingleGaussianAdapt(input_comp, alpha_params=array_alpha, rho_params=array_rho, im_show=False)
        else:
            matrix_mean, matrix_std, gt_test_comp = OneSingleGaussian(input, array_alpha, im_show=False)

        np.save("gt_test_comp_{}".format(OF_TYPE), gt_test_comp)

    else:
        gt_test_comp = np.load("gt_test_comp_{}".format(GAUSSIAN_METHOD))

    ##########################################################################
    ##########################################################################

    """
    ExtractPerformance
    """
    array_alpha = np.linspace(0, 5, 10, endpoint=True)
    # array_rho = np.array([0.22]) # Highway
    # array_rho = np.array([0.22]) #Fall
    array_rho = np.array([0.11])  # Traffic

    gt_test = np.load("gt_test_{}.npy".format(GAUSSIAN_METHOD))
    gt_test_comp = np.load("gt_test_comp_{}.npy".format(GAUSSIAN_METHOD))

    gt_train_comp = gt_test_comp[50:]

    TP_list, FP_list, TN_list, FN_list = extractPerformance_2Params(gt_train[:50], gt_test[:, :50],
                                                                    array_params_a=array_alpha, array_params_b=array_rho,
                                                                    im_show_performance=False)

    precision, recall, fscore, accuracy = metrics_2Params(TP_list, FP_list, TN_list, FN_list,
                                                                           array_params_a=array_alpha,
                                                                           array_params_b=array_rho)

    auc = getAUC(recall.reshape(len(array_alpha)), precision.reshape(len(array_alpha)))

    TP_list, FP_list, TN_list, FN_list = extractPerformance_2Params(gt_train_comp[50:], gt_test_comp[:, :50],
                                                                    array_params_a=array_alpha,
                                                                    array_params_b=array_rho,
                                                                    im_show_performance=False)

    precision_comp, recall_comp, fscore_comp, accuracy_comp = metrics_2Params(TP_list, FP_list, TN_list, FN_list,
                                                                              array_params_a=array_alpha,
                                                                              array_params_b=array_rho)

    auc_comp = getAUC(recall_comp.reshape(len(array_alpha)), precision_comp.reshape(len(array_alpha)))

    MakeYourGIF(gt_train_comp[50:], 'gt_train_comp.gif')
    MakeYourGIF(gt_test[np.argmax(fscore)], 'gt_test.gif')
    MakeYourGIF(gt_test_comp[np.argmax(fscore_comp)], 'gt_test_comp.gif')

    # Comparision of Plots
    plt.figure()
    plt.plot(recall.reshape(len(array_alpha)),
             precision.reshape(len(array_alpha)), 'g', label="Adaptative (AUC {0:.2f})".format(auc))
    plt.plot(recall_comp.reshape(len(array_alpha)),
             precision.reshape(len(array_alpha)), '--b', label="Adaptative + Compensation (AUC {0:.2f})".format(auc_comp))
    plt.legend(loc='lower left')
    plt.title("Precision-Recall")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("{}".format(OF_TYPE))
    plt.ylim([0, 1])
    # plt.xlim([0, 1])
    plt.savefig("baseline_vs_other_{}_{}_a.png".format(DATABASE, datetime.now().strftime('%d%m%y_%H-%M-%S')), bbox_inches='tight', frameon=False)
    plt.show()