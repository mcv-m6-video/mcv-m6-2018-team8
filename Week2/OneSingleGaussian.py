import numpy as np
import cv2
import matplotlib.pyplot as plt
import warnings

import sys


def OneSingleGaussian(input, array_alpha, im_show=True):

    print("\n ---------- OneSingleGaussian Method ---------- \n")
    # Training part, for training we only will use the 50% of the data
    data = input[0:len(input) // 2]
    rows, cols = input.shape[1:]

    #for k in range(len(input)/2):
        #matrixtofill = np.zeros((y, x))

    # matrix_mean = np.zeros((y,x))
    # matrix_std = np.zeros((y,x))
    # for i in range(y):
    #     for j in range(x):
    #         pixelval = input[0:(len(input)//2),i,j]
    #         pixelmean = np.mean(pixelval)
    #         pixelstd = np.std(pixelval)
    #         matrix_mean[i,j] = pixelmean
    #         matrix_std[i,j] = pixelstd

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
    print("\n")

    # for a, alpha in range(len(array_alpha)):
    #     gt_test_con = []
    #     # alpha = array_alpha[a]
    #     for k in range(len(data2)):
    #         gt_frame = np.zeros((y,x))
    #         for i in range(y):
    #             for j in range(x):
    #                 pixelval = data2[k,i,j]
    #                 if abs(pixelval - matrix_mean[i,j]) >= (alpha*(matrix_std[i,j] + 2)):
    #                     gt_frame[i,j] = 1
    #                 else:
    #                     gt_frame[i,j] = 0
    #         status = np.array_equal(gt_frame, gt_frame_v2[k])
    #         if not(status):
    #             warnings.warn("not equal")
    #
    #         print("{}/{} Status: {}".format(k,len(data2),status))
    #         gt_test_con.append(gt_frame)
    #     gt_test.append(gt_test_con)

    return matrix_mean, matrix_std, np.array(gt_test)