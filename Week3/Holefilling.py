import numpy as np
from scipy import ndimage

def Holefilling(input, connectivity, kernel=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])):

    print("Connectivity: {}".format(connectivity))

    if isinstance(input, (list,)):
        input = np.array(input)

    if input.dtype != np.uint8:
        input = np.uint8(input)

    gt_hole = []
    for i, img in enumerate(input):
        inside_mask = ndimage.binary_fill_holes(img*255, structure=kernel.astype(np.bool))
        # inside_mask = ndimage.binary_erosion(boolimage, structure=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).astype(np.bool))

        if connectivity == 4:
            el = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).astype(np.bool)
        else:
            el = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).astype(np.bool)

        outputImage = np.maximum(inside_mask, ndimage.grey_erosion(inside_mask, size=(7, 7), footprint=el))

        out = np.where(outputImage == True, 255, 0)

        gt_hole.append(out)

    return np.array(gt_hole)