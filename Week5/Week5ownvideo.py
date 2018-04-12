import sys
sys.path.append('../')
from common.config import *
from common.Database import *
from common.extractPerformance import *
from methods.GaussianMethods import *
from methods.AreaFiltering import *
from Week3.MorphologicTransformation import *
from Week3.Holefilling import *

from Tracking_KalmanFilter import *
from Tracking_CamShift import *
import cv2
import numpy as np

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

    # Crop region
    x1 = 400
    y1 = 600
    x2 = 1200
    y2 = 1900

    # Number of frames
    Nfr = 1000

    video = "VID_20180409_134421.mp4"
    cap = cv2.VideoCapture(video)

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    post_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    print("Frame: {}".format(post_frame))

    images_demo_color = []
    images_demo = []
    images_for_mean = []
    id = 0
    while (id<Nfr):

        status, frame = cap.read()

        if status:
            crop_img = frame[y1:y2, x1:x2]
            crop_img_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

            if id > 600 and id < 700:
                images_for_mean.append(crop_img_gray)

            images_demo.append(crop_img_gray)
            images_demo_color.append(crop_img)

            # im_test = crop_img.copy()
            # cv2.putText(im_test, "{}".format(id), (50, 50), cv2.FONT_HERSHEY_DUPLEX,
            #             fontScale=2, color=(0, 0, 255))
            # cv2.imshow("UAB", im_test)
            # cv2.waitKey(1)

        else:
            break

        sys.stdout.write("\r  {}/{}".format(id, Nfr))
        sys.stdout.flush()

        id+=1

    cap.release()
    cv2.destroyAllWindows()

    images_demo = np.array(images_demo)
    images_demo_color = np.array(images_demo_color)
    images_for_mean = np.array(images_for_mean)

    dir_to_save_fig = "output_images"
    dir_to_save_npy = "output_npy"
    if not os.path.exists(dir_to_save_fig):
        os.mkdir(dir_to_save_fig)

    if not os.path.exists(dir_to_save_npy):
        os.mkdir(dir_to_save_npy)

    # array_alpha = np.array([3.8])
    array_alpha = np.arange(0, 16, step=2)

    data = images_demo[240:600]
    if not os.path.exists(os.path.join(dir_to_save_npy, "gt_ownvideo_{}.npy".format(DATABASE))):

        matrix_mean, matrix_std, gt_test = OneSingleGaussian(data, array_alpha, num_of_train_images=0,
                                                             data_for_train=images_for_mean, im_show=False)

        np.save(os.path.join(dir_to_save_npy, "gt_ownvideo_{}".format(DATABASE)), gt_test)

    else:
        gt_test = np.load(os.path.join(dir_to_save_npy, "gt_ownvideo_{}.npy".format(DATABASE)))

    # for i, alpha in enumerate(array_alpha):
    #     for img in gt_test[i]:
    #         cv2.imshow("Alpha {}".format(alpha), cv2.convertScaleAbs(np.uint8(img), alpha=255.0))
    #         cv2.waitKey(10)
    #
    # cv2.destroyAllWindows()

    gt_test = gt_test[7]
    if use_morph_ex:
        kernel = cv2.getStructuringElement(MORPH_STRUCTURE, (3, 3))
        gt_test = MorphologicalTransformation(gt_test, kernel=kernel, type=MORPH_EX)

        kernel = cv2.getStructuringElement(MORPH_STRUCTURE, (3, 3))
        gt_test = MorphologicalTransformation(gt_test, kernel=kernel, type="erosion")

        gt_test = AreaFiltering(gt_test, 1000)
        gt_test = Holefilling(gt_test, connectivity=4, kernel=cv2.getStructuringElement(MORPH_STRUCTURE, (7, 7)))

    # for img in gt_test:
    #     cv2.imshow("Alpha {}".format(array_alpha[7]), cv2.convertScaleAbs(np.uint8(img), alpha=255.0))
    #     cv2.waitKey(10)
    #
    # cv2.destroyAllWindows()
    #
    # MakeYourGIF(cv2.convertScaleAbs(np.uint8(gt_test), alpha=255.0), "own_gif_mask.gif")

    if TRACKING_METHOD == "kalman":
        track_gt, speeds = Tracking_KalmanFilter(images_demo_color[240:600], gt_test, speed_estimator=8.5,
                                                 limit_speed=35, threshold_min_area=2500, debug=True)

    elif TRACKING_METHOD == "camshift":
        track_gt, speeds = Tracking_CamShift(images_demo_color[240:600], gt_test, speed_estimator=8.5,
                                             limit_speed=35, threshold_min_area=2500, debug=True)

    # Resize the viedo to have a lighter GIF (less Mb)
    # gt_test_resize = []
    # import scipy
    # for img in track_gt:
    #     rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     gt_test_resize.append(scipy.misc.imresize(rgb, 0.5))
    #
    # MakeYourGIF(gt_test_resize, "{}_ownvideo_tacking.gif".format(TRACKING_METHOD))








