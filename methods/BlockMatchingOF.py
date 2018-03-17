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
def BlockMatchingOpticalFlow(im1, im2, window_size=15, shift=3, stride=1):

    assert window_size % 2 == 1 and shift % 2 == 1

    # Initialize the matrices.
    vx = np.zeros( ((im2.shape[0] - window_size) //stride + 1, (im2.shape[1] - window_size) //stride + 1))
    vy = np.zeros( ((im2.shape[0] - window_size) // stride + 1, (im2.shape[1] - window_size) // stride + 1))
    wh = window_size // 2

    # Go through all the blocks.
    tx, ty = 0, 0
    for x in range(wh, im2.shape[0] - wh - 1, stride):
        for y in range(wh, im2.shape[1] - wh - 1, stride):
            nm = im2[x - wh:x + wh + 1, y - wh:y + wh + 1].flatten()

            min_dist = None
            flox, flowy = 0, 0
            # Compare each block of the next frame to each block from a greater
            # region with the same center in the previous frame.
            for i in range(max(x - shift, wh), min(x + shift + 1, im1.shape[0] - wh - 1)):
                for j in range(max(y - shift, wh), min(y + shift + 1, im1.shape[1] - wh - 1)):
                    om = im1[i - wh:i + wh + 1, j - wh:j + wh + 1].flatten()

                    # Compute the distance and update minimum.
                    dist = SSD(nm, om)
                    if not min_dist or dist < min_dist:
                        min_dist = dist
                        flowx, flowy = x - i, y - j

            # Update the flow field. Note the negative tx and the reversal of
            # flowx and flowy. This is done to provide proper quiver plots, but
            # should be reconsidered when using it.
            vx[-tx, ty] = flowy
            vy[-tx, ty] = flowx

            ty += 1
        tx += 1
        ty = 0

    return vx, vy


#
# Show the quiver plot of the 2D velocities.
#
# Input : The x- and y-velocities (both 2d numpy arrays).
# Output: The quiver plot (matplotlib figure).
#
def quiver_flow_field(vx, vy):
    # Generate the quiver plot.
    plt.figure()
    plt.quiver(vx, vy, color='r')
    plt.show()