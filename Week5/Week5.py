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

    dir_to_save_fig = "output_images"
    dir_to_save_npy = "output_npy"
    if not os.path.exists(dir_to_save_fig):
        os.mkdir(dir_to_save_fig)

    if not os.path.exists(dir_to_save_npy):
        os.mkdir(dir_to_save_npy)

    """
    AdaptativeGaussian Computation (no compensation)
    Extract the Foreground Image
    """

    # Highway
    array_alpha = np.array([3.8])
    array_rho = np.array([0.22])

    # array_rho = np.array([0.22]) #Fall
    # array_rho = np.array([0.11])  # Traffic

    if not os.path.exists(os.path.join(dir_to_save_npy, "gt_test_{}.npy".format(GAUSSIAN_METHOD))):

        if GAUSSIAN_METHOD in 'adaptative':
            matrix_mean, matrix_std, gt_test = OneSingleGaussianAdapt(input, alpha_params=array_alpha,
                                                                      rho_params=array_rho,
                                                                      num_of_train_images=len(input) // 2,
                                                                      im_show=False)
        else:
            matrix_mean, matrix_std, gt_test = OneSingleGaussian(input, array_alpha, im_show=False)

        np.save(os.path.join(dir_to_save_npy, "gt_test_{}".format(GAUSSIAN_METHOD)), gt_test)

    else:
        gt_test = np.load(os.path.join(dir_to_save_npy, "gt_test_{}.npy".format(GAUSSIAN_METHOD)))

    if use_morph_ex:
        kernel = cv2.getStructuringElement(MORPH_STRUCTURE, (3, 3))
        gt_test = MorphologicalTransformation(gt_test[0, 0], kernel=kernel, type=MORPH_EX)

    Tracking_KF(gt)