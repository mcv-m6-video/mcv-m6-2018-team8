import sys
import platform
sys.path.append('../.')
from common.config import *
from common.Database import *
from performanceW2 import *
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

    start_frame = 1050
    end_frame = 1350
    gt_db = Database(abs_dir_gt, start_frame=start_frame, end_frame=end_frame)
    input_db = Database(abs_dir_input, start_frame=start_frame, end_frame=end_frame)

    gt = gt_db.loadDB(im_color=False)
    input = input_db.loadDB(im_color=True)

    # Task3_ compare with different methods
    if isCV2():
        """
        BGSubstractorMOG
        """
        bg_substracttor = cv2.BackgroundSubtractorMOG()
        x_mask = extractBackgroundSubtrasctor_CV(bg_substracttor, data=input, im_show=False)
        print("\n\nBackgroundSubtractorMOG Method (from OpenCV {})".format(cv2.__version__))
        TP_list, FP_list, TN_list, FN_list = performanceW2(gt, x_mask)
        precision_list, recall_list, fscore_list, accuracy_list = metrics(TP_list, FP_list, TN_list, FN_list, x_mask)

        """
        BGSubstractorMOG2
        """
        bg_substracttor = cv2.BackgroundSubtractorMOG2()
        x_mask = extractBackgroundSubtrasctor_CV(bg_substracttor, data=input, im_show=False)
        print("\n\nBackgroundSubtractorMOG2 Method (from OpenCV {})".format(cv2.__version__))
        TP_list, FP_list, TN_list, FN_list = performanceW2(gt, x_mask)
        precision_list, recall_list, fscore_list, accuracy_list = metrics(TP_list, FP_list, TN_list, FN_list, x_mask)

    else:
        """
        BSubtractorGMG
        """
        print("\n\nBackgroundSubtractorGMG Method (from OpenCV {})".format(cv2.__version__))
        bg_substracttor = cv2.createBackgroundSubtractorGMG()
        x_mask = extractBackgroundSubtrasctor_CV(bg_substracttor, data=input, im_show=False)
        TP_list, FP_list, TN_list, FN_list = performanceW2(gt, x_mask)
        precision_list, recall_list, fscore_list, accuracy_list = metrics(TP_list, FP_list, TN_list, FN_list, x_mask)

    if platform.system() == 'Linux' and platform.machine() == 'x86_64':
        """
        SuBSENSE + LOBSTER: Only compiled for Linux x86_64 
        """
        print("\n\nSuBSENSE + LOBSTER Method (from ethereon's GitHub)")
        subsense = Subsense()
        x_mask = extractBackgroundSubtrasctor_CV(subsense, data=input, im_show=False)
        subsense.release()
        TP_list, FP_list, TN_list, FN_list = performanceW2(gt, x_mask)
        precision_list, recall_list, fscore_list, accuracy_list = metrics(TP_list, FP_list, TN_list, FN_list, x_mask)

