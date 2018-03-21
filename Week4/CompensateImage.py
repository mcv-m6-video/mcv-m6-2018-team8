import numpy as np
import cv2
from collections import Counter

def CompensateImage(prev_img, vx, vy, udir_accum=0, vdir_accum=0, block_size=None):

    rows, cols = prev_img.shape[:2]

    vx_nonzero = vx[vx!=0]
    vy_nonzero = vy[vy != 0]

    if len(vx_nonzero):
        ux, cx = np.unique(vx_nonzero, return_counts=True)
    else:
        ux, cx = np.unique(vx, return_counts=True)

    if len(vy_nonzero):
        uy, cy = np.unique(vy_nonzero, return_counts=True)
    else:
        uy, cy = np.unique(vy, return_counts=True)

    udir_accum += ux[np.argmax(cx)]
    vdir_accum += uy[np.argmax(cy)]

    # --- Other approaches
    # vx_hist = np.histogram(vx, bins=100)
    # vx_c, vx_n = vx_hist[0], vx_hist[1]

    # vy_hist = np.histogram(vx, bins=100)
    # vy_c, vy_n = vy_hist[0], vy_hist[1]

    # udir_accum += vx_n[np.argmax(vx_c)]
    # vdir_accum += vy_n[np.argmax(vy_c)]

    # --- Other approaches
    # vx2 = vx.flatten()
    # b = Counter(vx2)
    # b2 = b.most_common(1)
    # repx = b2[0][0]
    #
    # vy2 = vy.flatten()
    # b = Counter(vy2)
    # b2 = b.most_common(1)
    # repy = b2[0][0]
    #
    # udir_accum += repx
    # vdir_accum += repy
    
    M = np.float32([[1, 0, -udir_accum], [0, 1, -vdir_accum]])
    comp_img = cv2.warpAffine(prev_img, M, (cols,rows))
    
    return comp_img, udir_accum, vdir_accum