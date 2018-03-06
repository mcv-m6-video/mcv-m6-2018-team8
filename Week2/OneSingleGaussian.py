import numpy as np
import cv2
import sys
import warnings


def OneSingleGaussian(input, array_alpha, im_show=True):

    print("\n ---------- OneSingleGaussian Method ---------- ")
    # Training part, for training we only will use the 50% of the data
    data = input[0:len(input) // 2]
    # rows, cols = input.shape[1:]

    matrix_mean = np.mean(data,0).astype(np.float64)
    matrix_std = np.std(data, 0).astype(np.float64)

    if im_show:
        cv2.imshow("Mean Matrix", cv2.convertScaleAbs(matrix_mean, alpha=255 / np.max(matrix_mean)))
        cv2.imshow("Std Matrix", cv2.convertScaleAbs(matrix_std, alpha=255 / np.max(matrix_mean)))
        cv2.waitKey(0)

    data2 = input[(len(input) // 2):len(input)]

    gt_test =  []  # size -> ( len(array_alpha), len(data2), rows, cols )
    for alpha in array_alpha:
        sys.stdout.write("\r>  Computing for alpha = {:.2f}".format(alpha))
        sys.stdout.flush()
        gt_frames = np.where(np.abs(data2 - matrix_mean) >= (alpha * (matrix_std + 2)), 1, 0)
        gt_test.append(gt_frames)
    print("\n ---------------------------------------------- \n")

    return matrix_mean, matrix_std, np.array(gt_test)