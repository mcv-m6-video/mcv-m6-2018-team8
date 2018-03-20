import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2

"""
Sum of Squared Distance between two vectors
"""
def SSD(vector_a, vector_b):
    assert (len(vector_a) == len(vector_b))
    return np.sum(np.power( (vector_a - vector_b), 2))

#
# Perform block matching optical flow for two consecutive frames.
#
# Input : The two frames (both numpy arrays), the window size, the area around
#         the window in the previous frame, and the stride between two
#         estimations (all integers).
# Output: The x- and y-velocities (both numpy arrays).
#
"""
Comute the Block Matching Optical FLow algorithm using two consecutives frames from the Dataset
Return: X-Y directions
"""
def BlockMatchingOpticalFlow(im1, im2, window_size=15, shift=3, stride=1, motion_estimation="forward", im_show_debug=False):

    # only odd window_sizes (we want a center point)
    assert (window_size % 2 == 1)

    if motion_estimation == 'backward':
    # All pixels in the current image are associated to a pixel in the past image (but the contrary cannot be ensured).
        im_ref = im1 # past image
        im_study = im2 # current image
    elif motion_estimation == 'forward':
    # All pixels in the past image are associated to a pixel in the current image (but the contrary cannot be ensured).
        im_study = im1 # past image
        im_ref = im2 # current image
    else:
        raise ValueError("The name {} of the motion estimation does not exist".format(motion_estimation))

    # Initialize the matrices.
    # Size = W-F+2P / S + 1, where:
    # W-> image size
    # F-> windows size
    # P-> padding(1) or not(0)
    # S-> strie, step size
    vx = np.zeros( ((im2.shape[0] - window_size) //stride + 1, (im2.shape[1] - window_size) //stride + 1))
    vy = np.zeros( ((im2.shape[0] - window_size) // stride + 1, (im2.shape[1] - window_size) // stride + 1))
    half_window_size = window_size // 2

    # Go through all the blocks.
    tx, ty = 0, 0
    for row in range(half_window_size, im2.shape[0] - half_window_size - 1, stride):
        for col in range(half_window_size, im2.shape[1] - half_window_size - 1, stride):
            window_next_frame = im_ref[row - half_window_size:row + half_window_size + 1, col - half_window_size:col + half_window_size + 1]

            if im_show_debug:
                im_test_ref = im_ref.copy()
                # im_test_study = im_study.copy()

                cv2.putText(im_test_ref, "({},{})".format(col,row), (col, row),
                            cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255))
                cv2.rectangle(im_test_ref, (col - half_window_size, row - half_window_size),
                              (col + half_window_size, row + half_window_size), (255, 0, 0), thickness=1)
                cv2.imshow("Image Ref", im_test_ref)

            min_dist = None
            flowx, flowy = 0, 0
            # Compare each block of the next frame to each block from a greater
            # region with the same center in the previous frame.
            for r in range(max(row - shift, half_window_size), min(row + shift + 1, im1.shape[0] - half_window_size - 1)):
                for c in range(max(col - shift, half_window_size), min(col + shift + 1, im1.shape[1] - half_window_size - 1)):
                    window_previous_frame = im_study[r - half_window_size:r + half_window_size + 1, c - half_window_size:c + half_window_size + 1]

                    if im_show_debug:
                        im_test_study = im_study.copy()

                        cv2.rectangle(im_test_study, (c - half_window_size, r - half_window_size),
                                      (c + half_window_size, r + half_window_size), (0, 255, 0), thickness=1)
                        cv2.rectangle(im_test_study, (col - half_window_size, row - half_window_size),
                                      (col + half_window_size, row + half_window_size), (255, 0, 0), thickness=1)
                        cv2.imshow("Image Study", im_test_study)
                        cv2.waitKey(1)

                    # Compute the distance and update minimum.
                    dist = SSD(window_next_frame, window_previous_frame)

                    if min_dist is None or dist < min_dist:
                        # update values of min_dist and flowxy
                        min_dist = dist
                        flowx, flowy = row - r, col - c

            # Update the flow field. Note the negative tx and the reversal of
            # flowx and flowy. This is done to provide proper quiver plots, but
            # should be reconsidered when using it.
            vx[-tx, ty] = flowy
            vy[-tx, ty] = flowx
            ty += 1

        tx += 1
        ty = 0

    return vx, vy, np.array([vx, vy]).transpose(1,2,0)


"""
Generate the quiver plot (2D)
"""
def quiver_flow_field(vx, vy):
    plt.figure()
    plt.quiver(vx, vy, color='r')
    plt.show(1)