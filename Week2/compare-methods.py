import sys
import platform
sys.path.append('../.')
from common.config import *
from common.Database import *
# from performanceW2 import *
from common.extractPerformance import *
from metrics import *
from OneSingleGaussian import *
from thirdparty.Subsense import *
import cv2

def extractBackgroundSubtrasctor_CV(bg_method, data, im_show=False):

    x_mask = []
    i = 0
    while(i < len(data)):
        bg_mask = bg_method.apply(data[i])
        x_mask.append(bg_mask)

        if im_show:
            cv2.imshow("Original Frame", data[i])
            cv2.imshow("BackgroundSubstractorMOG2 Frame", bg_mask)
            cv2.waitKey(1)

        i += 1

    return x_mask

def isCV2():
    major = cv2.__version__.split(".")[0]
    return True if major == '2' else 0

if __name__ == "__main__":

    gt_db = Database(abs_dir_gt, start_frame=start_frame, end_frame=end_frame)
    input_db = Database(abs_dir_input, start_frame=start_frame, end_frame=end_frame)

    gt = gt_db.loadDB(im_color=False)
    input = input_db.loadDB(im_color=True)

    methods = ["MOG", "MOG2", "KNN", "Subsense", "Lobster"]
    SHOW_PLOT = True
    # Task3_ compare with different methods
    # if isCV2():
    """
    BGSubstractorMOG
    """
    params = np.linspace(1, 100, 20)
    x_mask = []
    print("\n\nBackgroundSubtractorMOG Method (from OpenCV {})".format(cv2.__version__))
    try:
        bg_substractor = cv2.bgsegm.createBackgroundSubtractorMOG()#BackgroundSubtractorMOG()
        for id, p in enumerate(params):
            bg_substracttor = cv2.bgsegm.createBackgroundSubtractorMOG() #TODO
            x_mask.append(extractBackgroundSubtrasctor_CV(bg_substractor, data=input, im_show=False))
            sys.stdout.write("\r>  Computing ... {:.2f}%".format((id + 1) * 100 / len(params)))
            sys.stdout.flush()

        print("\n")

        TP_list, FP_list, TN_list, FN_list = extractPerformance(gt, x_mask, array_params=params)
        precision_list, recall_list, fscore_list, accuracy_list = metrics(TP_list, FP_list, TN_list, FN_list, x_mask, array_params=params)

        if SHOW_PLOT:
            plotF1Score2D(np.linspace(0, params[-1], len(params)), fscore_list)
            plotPrecisionRecall(recall_list, precision_list, label=DATABASE)

    except AttributeError:
        warnings.warn("BackgroundSubtractorMOG does not exist in your OpenCV {}".format(cv2.__version__))

    """
    BGSubstractorMOG2
    """
    params = np.linspace(1, 100, 20)
    x_mask = []
    print("\n\nBackgroundSubtractorMOG2 Method (from OpenCV {})".format(cv2.__version__))
    try:
        bg_substractor = cv2.createBackgroundSubtractorMOG2()
        for id, p in enumerate(params):
            bg_substractor = cv2.createBackgroundSubtractorMOG2(history=0, varThreshold=p, detectShadows=False)
            x_mask.append(extractBackgroundSubtrasctor_CV(bg_substractor, data=input, im_show=False))
            sys.stdout.write("\r>  Computing ... {:.2f}%".format((id + 1) * 100 / len(params)))
            sys.stdout.flush()

        print("\n")

        TP_list, FP_list, TN_list, FN_list = extractPerformance(gt, x_mask, array_params=params)
        precision_list, recall_list, fscore_list, accuracy_list = metrics(TP_list, FP_list, TN_list, FN_list, x_mask, array_params=params)

        if SHOW_PLOT:
            plotF1Score2D(np.linspace(0, params[-1], len(params)), fscore_list)
            plotPrecisionRecall(recall_list, precision_list, label=DATABASE)

    except AttributeError:
        warnings.warn("BackgroundSubtractorMOG2 does not exist in your OpenCV {}".format(cv2.__version__))

    """
    BSubtractorKNN
    """
    params = np.linspace(1, 100, 20)
    x_mask = []
    print("\n\nBackgroundSubtractorKNN Method (from OpenCV {})".format(cv2.__version__))
    try:
        bg_substractor = cv2.BackgroundSubtractorKNN()
        for id, p in enumerate(params):
            bg_substractor = cv2.createBackgroundSubtractorKNN(history=0, dist2Threshold=p, detectShadows=False)
            x_mask.append(extractBackgroundSubtrasctor_CV(bg_substractor, data=input, im_show=False))
            sys.stdout.write("\r>  Computing ... {:.2f}%".format((id + 1) * 100 / len(params)))
            sys.stdout.flush()

        print("\n")

        TP_list, FP_list, TN_list, FN_list = extractPerformance(gt, x_mask, array_params=params)
        precision_list, recall_list, fscore_list, accuracy_list = metrics(TP_list, FP_list, TN_list, FN_list, x_mask, array_params=params)

        if SHOW_PLOT:
            plotF1Score2D(np.linspace(0, params[-1], len(params)), fscore_list)
            plotPrecisionRecall(recall_list, precision_list, label=DATABASE)

    except AttributeError:
        warnings.warn("BackgroundSubtractorKNN does not exist in your OpenCV {}".format(cv2.__version__))

    if platform.system() == 'Linux' and platform.machine() == 'x86_64':
        """
        SuBSENSE + LOBSTER: Only compiled for Linux x86_64 
        """
        print("\n\nSuBSENSE + LOBSTER Method (from ethereon's GitHub)")
        subsense = Subsense()
        x_mask = extractBackgroundSubtrasctor_CV(subsense, data=input, im_show=False)
        subsense.release()
        TP_list, FP_list, TN_list, FN_list = extractPerformance(gt, x_mask)
        precision_list, recall_list, fscore_list, accuracy_list = metrics(TP_list, FP_list, TN_list, FN_list, x_mask)

        subsense = Lobster()
        x_mask = extractBackgroundSubtrasctor_CV(subsense, data=input, im_show=False)
        subsense.release()
        TP_list, FP_list, TN_list, FN_list = extractPerformance(gt, x_mask)
        precision_list, recall_list, fscore_list, accuracy_list = metrics(TP_list, FP_list, TN_list, FN_list, x_mask)