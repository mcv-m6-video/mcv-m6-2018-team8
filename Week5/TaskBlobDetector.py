from common.config import *
from common.Database import *
from methods.GaussianMethods import *
from Week3.MorphologicTransformation import *
sys.path.append('../')

if __name__ == "__main__":

    gt_db = Database(abs_dir_gt, start_frame=start_frame, end_frame=end_frame)
    input_db = Database(abs_dir_input, start_frame=start_frame, end_frame=end_frame)

    gt = gt_db.loadDB(im_color=cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
    input = input_db.loadDB(im_color=False)

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
                                                                      rho_params=array_rho, num_of_train_images=len(input)//2, im_show=False)
        else:
            matrix_mean, matrix_std, gt_test = OneSingleGaussian(input, array_alpha, im_show=False)

        np.save(os.path.join(dir_to_save_npy, "gt_test_{}".format(GAUSSIAN_METHOD)), gt_test)

    else:
        gt_test = np.load(os.path.join(dir_to_save_npy, "gt_test_{}.npy".format(GAUSSIAN_METHOD)))

    if use_morph_ex:
        kernel = cv2.getStructuringElement(MORPH_STRUCTURE, (3, 3))
        gt_test = MorphologicalTransformation(gt_test[0,0], kernel=kernel, type=MORPH_EX)

    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 0.1
    params.maxThreshold = 1000

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 500

    # Filter by Circularity
    params.filterByCircularity = False
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    is_cv3 = cv2.__version__.startswith("3.")
    if is_cv3:
        detector = cv2.SimpleBlobDetector_create(params)
    else:
        detector = cv2.SimpleBlobDetector(params)

    # Detect blobs.
    for id, frame in enumerate(gt_test):
        frame_uint8 = cv2.cvtColor(np.uint8(frame*255), cv2.COLOR_GRAY2BGR)
        keypoints = detector.detect(frame_uint8)

        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        im_with_keypoints = cv2.drawKeypoints(frame_uint8, keypoints, np.array([]), (0, 0, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Show keypoints
        cv2.imshow("Keypoints", im_with_keypoints)
        cv2.waitKey(1)