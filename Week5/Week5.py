from common.config import *
from common.Database import *
from methods.GaussianMethods import *
from Week3.MorphologicTransformation import *
sys.path.append('../')

from Tracking_KF import *

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

    gt = gt_db.loadDB(im_color=cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
    input = input_db.loadDB(im_color=True)

    # array_rho = np.array([0.22])  # 0.11 for traffic
    # array_alpha = np.linspace(0, 5, 10, endpoint=True)
    #
    # gt_test = []
    # if GAUSSIAN_METHOD in 'adaptative':
    #     matrix_mean, matrix_std, gt_test = OneSingleGaussianAdapt(input, alpha_params=array_alpha,
    #                                                               rho_params=array_rho,
    #                                                               im_show=False) if not checkFilesNPY("numpy_files") else checkFilesNPY("numpy_files")
    # else:
    #     matrix_mean, matrix_std, gt_test = OneSingleGaussian(input, array_alpha, im_show=False)


    # gt_test = np.load("gt_test_highway.npy")
    # gt_test = np.where(gt_test==True,255,0)

    Tracking_KF(gt)
    # filtered_state_means, filtered_state_covariances = Tracking_KF(gt)