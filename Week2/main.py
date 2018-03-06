import sys
sys.path.append('../.')
from common.config import *
from common.Database import *
from OneSingleGaussian import *
from performanceW2 import *
from metrics import *
import matplotlib.pyplot as plt
from sklearn.metrics import auc

if __name__ == "__main__":

    start_frame = 1050
    end_frame = 1350
    gt_db = Database(abs_dir_gt, start_frame=start_frame, end_frame=end_frame)
    input_db = Database(abs_dir_input, start_frame=start_frame, end_frame=end_frame)
    results_db = Database(abs_dir_result, start_frame=0)

    gt = gt_db.loadDB(im_color=False)
    input = input_db.loadDB(im_color=False)
    #res = results_db.loadDB(im_color=False)

    #res_A = res[:200] #testA
    #res_B = res[200:] #testB

    array_alpha = np.arange(0,10,0.2)
    matrix_mean, matrix_std, gt_test = OneSingleGaussian(input, array_alpha, im_show=False)

    # for idx, a in enumerate(array_alpha):
    #     cv2.imshow("GT with alpha {}".format(a), gt_test[idx][0])
    #
    # cv2.waitKey(0)

    # cv2.imwrite("matrix_mean_fall2.png", matrix_mean)
    # cv2.imshow("Matrix Mean", matrix_mean)
    # cv2.imwrite("matrix_std_fall2.png", matrix_std)
    # cv2.imshow("Matrix Std", matrix_std)

    gt2 = gt[(len(gt)/2):]
    TP_list, FP_list, TN_list, FN_list = performanceW2(gt2, gt_test, array_params=array_alpha)
    precision_list, recall_list, fscore_list, accuracy_list = metrics(TP_list, FP_list, TN_list, FN_list, gt_test, array_params=array_alpha)
    area_auc = auc(recall_list, precision_list)

    x = np.linspace(0, array_alpha[-1], len(array_alpha))
    plt.figure(1)
    plt.plot(x, fscore_list, 'b', label='F1-score')
    plt.legend(loc='lower right')
    plt.xlabel("Alpha")
    plt.ylabel("F1-score")
    plt.axis([0, max(array_alpha), 0, max(fscore_list)+0.2])
    plt.show()

    # plt.step(recall_list, precision_list, color='b', alpha=0.2, where='post')
    # plt.fill_between(recall_list, precision_list, step='post', alpha=0.2, color='b')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.ylim([0.0, 1.05])
    # plt.xlim([0.0, 1.0])
    # plt.title('Precision-Recall curve: AP={0:0.2f}'.format(average_precision))

    plt.figure()
    plt.plot(recall_list, precision_list, 'r', label='Highway')
    plt.axis([0, 1, 0, 1])
    plt.title("Precision-Recall")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc='upper right')
    plt.show()