import numpy as np
from scipy import ndimage

def Holefilling(gt_test,connectivity):

    print "Connectivity:", connectivity
    gt_hole = []
    for t in range(len(gt_test)):

        gt_test2 = gt_test[t,:,:,:]
        numimag = gt_test2.shape[0]
        edgelist = []

        for i in range(numimag):
            boolimage = gt_test2[i,:,:]
            intimage = 255*boolimage
            h_max = np.max(intimage)

            inside_mask = ndimage.binary_fill_holes(boolimage, structure=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).astype(np.bool))
            # inside_mask = ndimage.binary_erosion(boolimage, structure=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).astype(np.bool))

    #        edge_mask = (boolimage & ~inside_mask)

            outputImage = np.copy(intimage)
            outputImage[inside_mask] = h_max

            if connectivity == 4:
                el = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).astype(np.bool)
            else:
                el = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).astype(np.bool)

            outputImage = np.maximum(intimage, ndimage.grey_erosion(outputImage, size=(3, 3), footprint=el))

            out = np.where(outputImage == 255, 1, 0)

            edgelist.append(out)

        gt_hole.append(edgelist)

    return gt_hole