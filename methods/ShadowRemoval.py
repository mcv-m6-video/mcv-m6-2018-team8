import numpy as np
import cv2


def normImage(im):
    max = 0.001 if np.max(im) == 0 else np.max(im)
    im = cv2.convertScaleAbs(im, alpha=255 / max)
    return im

"""
Idea from SHADOW REMOVAL WITH BLOB-BASED MORPHOLOGICAL RECONSTRUCTION FOR ERROR CORRECTION: Li-Qun Xu, José Luis Landabaso, Montse Pardàs

Get a normalised chromatic colour space (brightness and chromacity):
r = R/(R+G+B)
g = G/(R+G+B)
"""
def BGR2RGS(images, im_show=False):
    # This is an implementation of the rgs method described by Elgammal et al.
    # in "Non-parametric model for background subtraction." Computer Vision 751-767.
    # It expects to receive a BGR image.

    if images.ndim == 4:
        B = images[:,:,:,0]
        G = images[:,:,:,1]
        R = images[:,:,:,2]
    elif images.ndim == 3:
        B = images[:, :, 0]
        G = images[:, :, 1]
        R = images[:, :, 2]
    else:
        raise ValueError("images.ndim == 3 || images.ndim == 4 ")

    # The luminance will be the sum of all channels
    sum_channels = B + G + R

    # Normalized chromacity coordinates
    with np.errstate(divide='ignore', invalid='ignore'):
        gp = np.float32(np.true_divide(G, sum_channels))
        rp = np.float32(np.true_divide(R, sum_channels))
        gp[gp == np.inf] = 0
        rp[rp == np.inf] = 0
        gp = np.nan_to_num(gp)
        rp = np.nan_to_num(rp)

    # gp = np.float32(np.divide(G, sum_channels))
    # rp = np.float32(np.divide(R,sum_channels + 0.001))

    # Brightness
    I = np.float32(np.divide(sum_channels, 3))

    rgs = np.array([I, gp, rp], dtype=np.uint8)
    RGS = np.array([I, G, R], dtype=np.uint8)

    if images.ndim == 4:
        # number_of_images, rows,cols,ch
        rgs = rgs.transpose(1, 2, 3, 0)
        RGS = RGS.transpose(1, 2, 3, 0)
    elif images.ndim == 3:
        # rows,cols,ch
        rgs = rgs.transpose(1, 2, 0)
        RGS = RGS.transpose(1, 2, 0)
    if im_show:
        for i in range(len(I)):
            cv2.imshow("I", I[i].astype(np.uint8))
            cv2.imshow("G", gp[i].astype(np.uint8))
            cv2.imshow("R", rp[i].astype(np.uint8))
            cv2.imshow("RGS", RGS[i])
            cv2.imshow("rgs", rgs[i])
            cv2.waitKey(100)

        cv2.destroyAllWindows()

    return RGS

def ShadowRemoval(input_color, gt_test, array_alpha, array_rho, im_show=False):

    assert(len(array_alpha)*len(array_rho) == len(gt_test))

    RGS = BGR2RGS(input_color, im_show=False)

    gt_shadow = np.zeros((gt_test[0].shape), dtype=np.bool)

    scalar_RGS = np.sqrt(np.power(RGS[:,:,:,0],2) + np.power(RGS[:,:,:,1],2) + np.power(RGS[:,:,:,2],2))

    gt_test_shadow = []
    total_id = 0
    for id_a, alpha in enumerate(array_alpha, start=0):
        for id_r, rho in enumerate(array_rho):
            RGS_Fore = np.where(gt_test[total_id] == True, scalar_RGS, 0)
            RGS_Back = np.where(gt_test[total_id] == False, scalar_RGS, 0)

            gt_uint8 = np.uint8(gt_test[total_id])*255 # [0,1] -> [0,255]

            BD = np.power((RGS_Fore - 0.8*RGS_Back), 2)
            chromatic = np.float32(np.multiply(BD,RGS_Back))
            CD = np.sqrt(np.power((RGS_Fore-0.8*chromatic),2))

            BD_background = CD < 10
            BD_shadow = np.bitwise_and(BD >= 0.5, BD <= 1.0)
            BD_high = np.bitwise_and(BD > 1.5, BD <= 1.25)
            BD_shadow = np.bitwise_and(BD_background, BD_shadow)
            BD_high = np.bitwise_and(BD_background, BD_high)

            gt_uint8 = np.uint8(np.where(BD_shadow, 50, gt_uint8)) # shadow
            gt_shadow =  np.where(BD_shadow, False, gt_test[total_id]) # shadow
            gt_uint8 = np.uint8(np.where(BD_high, 80, gt_uint8)) # highlight
            gt_uint8[gt_test[total_id] == False] = 0 # only on the Foreground objects
            gt_shadow[gt_test[total_id] == False] = 0  # only on the Foreground objects

            if im_show:
                for i in range(len(RGS_Fore)):
                    gt_color_uint8 = cv2.cvtColor(gt_uint8[i], cv2.COLOR_GRAY2BGR)
                    gt_color_uint8[:, :, 0] = np.where(gt_color_uint8[:, :, 0] == 50, 0, gt_color_uint8[:,:,0])
                    gt_color_uint8[:, :, 1] = np.where(gt_color_uint8[:, :, 1] == 50, 0, gt_color_uint8[:,:,1])
                    gt_color_uint8[:, :, 2] = np.where(gt_color_uint8[:, :, 2] == 50, 255, gt_color_uint8[:,:,2])

                    gt_color_uint8[:, :, 0] = np.where(gt_color_uint8[:, :, 0] == 80, 0, gt_color_uint8[:, :, 0])
                    gt_color_uint8[:, :, 1] = np.where(gt_color_uint8[:, :, 1] == 80, 255, gt_color_uint8[:, :, 1])
                    gt_color_uint8[:, :, 2] = np.where(gt_color_uint8[:, :, 2] == 80, 0, gt_color_uint8[:, :, 2])

                    gt_diff = cv2.absdiff(gt_test[total_id,i].astype(np.uint8)*255, gt_uint8[i])

                    cv2.imshow("BD", cv2.convertScaleAbs(BD[i].astype(np.uint8), alpha=255 / np.max(BD[i].astype(np.uint8))))
                    cv2.imshow("CD", cv2.convertScaleAbs(CD[i].astype(np.uint8), alpha=255 / np.max(CD[i].astype(np.uint8))))
                    cv2.imshow("Fore", cv2.convertScaleAbs(RGS_Fore[i].astype(np.uint8), alpha=255 / np.max(RGS_Fore[i].astype(np.uint8))))
                    cv2.imshow("Back", cv2.convertScaleAbs(RGS_Back[i].astype(np.uint8), alpha=255 / np.max(RGS_Back[i].astype(np.uint8))))
                    cv2.imshow("Mask", np.uint8(gt_test[total_id,i])*255) # [0,1] -> [0,255]
                    cv2.imshow("Mask shadow removal", gt_color_uint8)
                    cv2.imshow("Mask Diff", gt_diff)
                    cv2.imshow("Input", input_color[i])
                    cv2.waitKey(100)

                cv2.destroyAllWindows()

            total_id += 1

        gt_test_shadow.append(gt_shadow)

    return np.array(gt_test_shadow)