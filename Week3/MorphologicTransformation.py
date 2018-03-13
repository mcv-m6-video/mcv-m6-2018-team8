import cv2
import numpy as np

"""
MorphologicalTransformation
@input: image or images to tranform
@kernel: structuring element to do the Morphological transofration
@type 'erosion', 'dilation', 'opening', 'closing', gradient', 'top-hat', 'black-hat'
"""
def MorphologicalTransformation(input, kernel, type):

    if isinstance(input, (list,)):
        input = np.array(input)

    if input.dtype == 'bool':
        input = input.astype(np.uint8)

    list_transf = []
    for i, img in enumerate(input):

        if type == 'erosion':
            transf = cv2.erode(img, kernel, iterations=1)
        elif type == 'dilation':
            transf = cv2.dilate(img, kernel, iterations=1)
        elif type == 'opening':
            transf = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        elif type == 'closing':
            transf = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        elif type == 'gradient':
            transf = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
        elif type == 'top-hat':
            transf = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
        elif type == 'black-hat':
            transf = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
        else:
            raise ValueError("{} does not exist as Morphologycal Transformation". format(type))

        list_transf.append(transf)

    return np.array(list_transf)