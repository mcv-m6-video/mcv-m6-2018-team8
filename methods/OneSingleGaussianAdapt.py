import numpy as np
import sys
import cv2

def OneSingleGaussianAdapt(input, alpha_params, rho_params, im_show=True):

    print("\n ---------- OneSingleGaussianAdaptative Method ---------- \n")
    # Training part, for training we only will use the 50% of the data
    data = input[:(len(input) // 2)]
    rows, cols = input.shape[1:]

    matrix_mean = np.mean(data,0).astype(np.float64)
    matrix_std = np.std(data, 0).astype(np.float64)

    if im_show:
        cv2.imshow("Mean Matrix", cv2.convertScaleAbs(matrix_mean, alpha=255 / np.max(matrix_mean)))
        cv2.imshow("Std Matrix", cv2.convertScaleAbs(matrix_std, alpha=255 / np.max(matrix_mean)))
        cv2.waitKey(0)

    data2 = input[(len(input) // 2):]

    gt_test = []
    # print("Computing Method A")

    matrix_mean_array = np.zeros((len(data2), rows, cols))
    matrix_std_array = np.zeros((len(data2), rows, cols))
    for id in range(len(data2)):
        matrix_mean_array[id] = matrix_mean
        matrix_std_array[id] = matrix_std

    total_id = 0
    for id_a, alpha in enumerate(alpha_params):
        for id_r, rho in enumerate(rho_params):
            gt_frames = np.zeros((len(data2), rows, cols), dtype=np.bool)

            sys.stdout.write("\r>  Computing for alpha = {:.2f} and rho = {:.2f} ... {}%".format(alpha, rho, (total_id)*100 / (len(alpha_params)*len(rho_params))))

            for k in range(len(data2)):
                for r in range(rows):
                    for c in range(cols):
                        if np.abs(data2[k,r,c] - matrix_mean[r,c]) >= (alpha * (matrix_std[r,c] + 2)):
                            gt_frames[k,r,c] = 1
                        else:
                            matrix_mean[r,c] = rho * data2[k,r,c] + (1 - rho) * matrix_mean[r,c]
                            matrix_std[r,c] = np.sqrt(rho * np.float_power(data2[k,r,c] - matrix_mean[r,c], 2) + np.float_power((1 - rho) * matrix_std[r,c],2))
                            gt_frames[k,r,c] = 0

            total_id += 1

            sys.stdout.write("\r>  Computing for alpha = {:.2f} and rho = {:.2f} ... {}%".format(alpha, rho, (total_id) * 100 / (len(alpha_params) * len(rho_params))))
            sys.stdout.flush()

            gt_test.append(gt_frames)

    print("\n -------------------------------------------------------- \n")

    return matrix_mean, matrix_std, np.array(gt_test)