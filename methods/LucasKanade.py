import cv2
import numpy as np

import numpy as np
import scipy.signal as si
from PIL import Image
import matplotlib.pyplot as plt
import scipy.signal as si
from PIL import Image
#import deriv
import numpy.linalg as lin
from scipy import signal

def lucas_kanade_np(prvs, next, win=2):
    assert prvs.shape == next.shape

    im1 = cv2.cvtColor(prvs, cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)

    I_x = np.zeros(im1.shape)
    I_y = np.zeros(im1.shape)
    I_t = np.zeros(im1.shape)
    I_x[1:-1, 1:-1] = (im1[1:-1, 2:] - im1[1:-1, :-2]) / 2
    I_y[1:-1, 1:-1] = (im1[2:, 1:-1] - im1[:-2, 1:-1]) / 2
    I_t[1:-1, 1:-1] = im1[1:-1, 1:-1] - im2[1:-1, 1:-1]
    params = np.zeros(im1.shape + (5,)) #Ix2, Iy2, Ixy, Ixt, Iyt
    params[..., 0] = I_x * I_x # I_x2
    params[..., 1] = I_y * I_y # I_y2
    params[..., 2] = I_x * I_y # I_xy
    params[..., 3] = I_x * I_t # I_xt
    params[..., 4] = I_y * I_t # I_yt
    del I_x, I_y, I_t
    cum_params = np.cumsum(np.cumsum(params, axis=0), axis=1)
    del params
    win_params = (cum_params[2 * win + 1:, 2 * win + 1:] -
                  cum_params[2 * win + 1:, :-1 - 2 * win] -
                  cum_params[:-1 - 2 * win, 2 * win + 1:] +
                  cum_params[:-1 - 2 * win, :-1 - 2 * win])
    del cum_params
    op_flow = np.zeros(im1.shape + (2,))
    det = win_params[...,0] * win_params[..., 1] - win_params[..., 2] **2
    op_flow_x = np.where(det != 0, (win_params[..., 1] * win_params[..., 3] - win_params[..., 2] * win_params[..., 4]) / det, 0)
    op_flow_y = np.where(det != 0, (win_params[..., 0] * win_params[..., 4] -  win_params[..., 2] * win_params[..., 3]) / det,0)
    op_flow[win + 1: -1 - win, win + 1: -1 - win, 0] = op_flow_x[:-1, :-1]
    op_flow[win + 1: -1 - win, win + 1: -1 - win, 1] = op_flow_y[:-1, :-1]

    return np.array(op_flow)

def LucasKanade(im1, im2):

    feature_params = dict(maxCorners = im1.shape[0]*im1.shape[1], qualityLevel = 0.1, minDistance = 5, blockSize = 7)
    lk_params = dict(winSize  = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    f1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    f2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(f1, mask = None, **feature_params)
    p1, st, err = cv2.calcOpticalFlowPyrLK(f1, f2,  p0, None, **lk_params)

    return p0, p1, st, err

def gauss_kern():
   h1 = 15
   h2 = 15
   x, y = np.mgrid[0:h2, 0:h1]
   x = x-h2/2
   y = y-h1/2
   sigma = 1.5
   g = np.exp( -( x**2 + y**2 ) / (2*sigma**2) );
   return g / g.sum()

def deriv(im1, im2):
   g = gauss_kern()
   Img_smooth = si.convolve(im1,g,mode='same')
   fx,fy=np.gradient(Img_smooth)
   ft = si.convolve2d(im1, 0.25 * np.ones((2,2))) + \
       si.convolve2d(im2, -0.25 * np.ones((2,2)))

   fx = fx[0:fx.shape[0]-1, 0:fx.shape[1]-1]
   fy = fy[0:fy.shape[0]-1, 0:fy.shape[1]-1];
   ft = ft[0:ft.shape[0]-1, 0:ft.shape[1]-1];

   return fx, fy, ft


def lk(im1, im2, i, j, window_size) :
   fx, fy, ft = deriv(im1, im2)
   halfWindow = np.floor(window_size/2)
   curFx = fx[i-halfWindow-1:i+halfWindow,
              j-halfWindow-1:j+halfWindow]
   curFy = fy[i-halfWindow-1:i+halfWindow,
              j-halfWindow-1:j+halfWindow]
   curFt = ft[i-halfWindow-1:i+halfWindow,
              j-halfWindow-1:j+halfWindow]
   curFx = curFx.T
   curFy = curFy.T
   curFt = curFt.T

   curFx = curFx.flatten(order='F')
   curFy = curFy.flatten(order='F')
   curFt = -curFt.flatten(order='F')

   A = np.vstack((curFx, curFy)).T
   U = np.dot(np.dot(lin.pinv(np.dot(A.T,A)),A.T),curFt)
   return U[0], U[1]




def LKVid(cap):

    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0,255,(100,3))

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    while(1):
        ret,frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]

        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
        img = cv2.add(frame,mask)

        #cv2.imshow('frame',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)

    cv2.destroyAllWindows()
    cap.release()

    return p0