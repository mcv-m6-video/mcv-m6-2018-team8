from skimage import morphology
from skimage import *
import numpy as np

def AreaFiltering(input,area):

    # if isinstance(input, (list,)):
    #     input = np.array(input)

    area_filtered = []
    for id, images in enumerate(input):
        out = morphology.remove_small_objects(images, area, connectivity=1)
        area_filtered.append(out)

    # for k in range(len(gt_test)):
    #     gt_test2 = gt_test[k]
    #     numimag = gt_test2.shape[0]
    #     arealist = []
    #     for i in range(0,numimag):
    #         boolimage = gt_test2[i]
    #         out = morphology.remove_small_objects(boolimage, area2,connectivity=2)
    #         arealist.append(out)
    #     Areafiltered.append(arealist)


#    gt_test2 = gt_test[0,:,:,:]
#    numimag = gt_test2.shape[0]
#
#    arealist = []
#    Areafiltered =[]
#    for p in area_array:
#        out = morphology.remove_small_objects(boolimage, p,connectivity=2)
#        arealist.append(out)
#        arealist = []
#        for i in range(0,numimag):
#            boolimage = gt_test2[i,:,:]
##            intimage = 255*boolimage
#            Areafiltered.append(arealist)


    # Areafilteredord = np.einsum('klij->lkij', Areafiltered)


    return np.array(area_filtered)