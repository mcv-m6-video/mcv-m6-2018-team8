import cv2
import sys, warnings
import numpy as np


def OneSingleGaussian(input, array_alpha, im_show=True):

    print("\n ---------- OneSingleGaussian Method ---------- \n")

    # Training part, for training we only will use the 50% of the data
    data = input[0:len(input) // 2]

    matrix_mean = np.mean(data,0).astype(np.float64)
    matrix_std = np.std(data, 0).astype(np.float64)

    if im_show:
        cv2.imshow("Mean Matrix", cv2.convertScaleAbs(matrix_mean, alpha=255 / np.max(matrix_mean)))
        cv2.imshow("Std Matrix", cv2.convertScaleAbs(matrix_std, alpha=255 / np.max(matrix_mean)))
        cv2.waitKey(0)

    data2 = input[(len(input) // 2):len(input)]

    gt_test =  []  # size -> ( len(array_alpha), len(data2), rows, cols )
    for i, alpha in enumerate(array_alpha):
        sys.stdout.write("\r>  Computing for alpha = {:.2f} ... {}%".format(alpha, i*100/len(array_alpha)))
        sys.stdout.flush()
        gt_frames = np.where(np.abs(data2 - matrix_mean) >= (alpha * (matrix_std + 2)), 1, 0)
        gt_test.append(gt_frames)
    print("\n ---------------------------------------------- \n")

    return matrix_mean, matrix_std, np.array(gt_test)

def OneSingleGaussianAdapt(input, alpha_params, rho_params, num_of_train_images=None, im_show=True):

    print("\n ---------- OneSingleGaussianAdaptative Method ---------- \n")

    if num_of_train_images is None:
        num_of_train_images = len(input)
    else:
        # at least one image for test
        assert (num_of_train_images < len(input))

    # Only use the training data selected
    train_data = input[:num_of_train_images]
    rows, cols = input.shape[1:]

    matrix_mean = np.mean(train_data, 0).astype(np.float64)
    matrix_std = np.std(train_data, 0).astype(np.float64)

    if im_show:
        cv2.imshow("Mean Matrix", cv2.convertScaleAbs(matrix_mean, alpha=255 / np.max(matrix_mean)))
        cv2.imshow("Std Matrix", cv2.convertScaleAbs(matrix_std, alpha=255 / np.max(matrix_mean)))
        cv2.waitKey(0)

    test_data = input[num_of_train_images:]

    matrix_mean_array = np.zeros((len(test_data), rows, cols))
    matrix_std_array = np.zeros((len(test_data), rows, cols))
    for id in range(len(test_data)):
        matrix_mean_array[id] = matrix_mean
        matrix_std_array[id] = matrix_std

    # gt_all_test = []
    gt_all_test = np.zeros((len(alpha_params), len(rho_params), len(test_data), rows, cols))
    gt_frames = np.zeros((len(test_data), rows, cols), dtype=np.bool)

    total_id = 0
    for id_a, alpha in enumerate(alpha_params):
        for id_r, rho in enumerate(rho_params):
            sys.stdout.write("\r>  Computing for alpha = {:.2f} and rho = {:.2f} ... {}%\n".format(alpha, rho, (total_id)*100 / (len(alpha_params)*len(rho_params))))

            for k in range(len(test_data)):
                for r in range(rows):
                    for c in range(cols):
                        if np.abs(test_data[k,r,c] - matrix_mean[r,c]) >= (alpha * (matrix_std[r,c] + 2)):
                            gt_frames[k,r,c] = 1
                        else:
                            matrix_mean[r,c] = rho * test_data[k,r,c] + (1 - rho) * matrix_mean[r,c]
                            matrix_std[r,c] = np.sqrt(rho * np.float_power(test_data[k,r,c] - matrix_mean[r,c], 2) + np.float_power((1 - rho) * matrix_std[r,c],2))
                            gt_frames[k,r,c] = 0

                sys.stdout.write("\r   {:.2f}%".format((k+1) * 100 / (len(test_data))))
                sys.stdout.flush()

            total_id += 1

            gt_all_test[id_a, id_r, :] = gt_frames
            # gt_all_test.append(gt_frames)

    print("\n -------------------------------------------------------- \n")

    return matrix_mean, matrix_std, np.array(gt_all_test)