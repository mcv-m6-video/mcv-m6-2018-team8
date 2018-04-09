import numpy as np
from scipy import ndimage

def Holefilling(input, connectivity):

    print ("Connectivity: {}".format(connectivity))

    gt_hole = []
    for i, img in enumerate(input):
        inside_mask = ndimage.binary_fill_holes(img*255, structure=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).astype(np.bool))
        # inside_mask = ndimage.binary_erosion(boolimage, structure=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).astype(np.bool))

        if connectivity == 4:
            el = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).astype(np.bool)
        else:
            el = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).astype(np.bool)

        outputImage = np.maximum(inside_mask, ndimage.grey_erosion(inside_mask, size=(3, 3), footprint=el))

        out = np.where(outputImage == True, 255, 0)

        gt_hole.append(out)

    return gt_hole