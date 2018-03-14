from skimage import morphology
from skimage import *
import numpy as np

def AreaFiltering(input,area):

    if isinstance(input, (list,)):
        input = np.array(input)

    area_filtered = []
    for id, images in enumerate(input):
        out = morphology.remove_small_objects(images, area, connectivity=1)
        area_filtered.append(out)

    return np.array(area_filtered)