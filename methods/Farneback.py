import cv2

def Farneback(im1, im2, win=7):

    assert (im1.shape == im2.shape)

    if im1.ndim == 3:
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)

    if im2.ndim == 3:
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(im1, im2, None, 0.5, 3, win, 3, 5, 1.2, 0)
    # flow = cv2.calcOpticalFlowFarneback(prvs, next, 0.5, 5, 15, 9, 7, 1.5, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

    return flow