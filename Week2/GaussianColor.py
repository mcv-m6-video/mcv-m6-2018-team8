import numpy as np
import sys

def GaussianColorRGB(input, array_alpha):

    print("\n ---------- Gaussian RGB Method ---------- \n")
    # Training part, for training we only will use the 50% of the data
    data = input[0:len(input) // 2]

    datablue = data[:,:,:,0]
    datagreen = data[:,:,:,1]
    datared = data[:,:,:,2]

    matrix_meanb = np.mean(datablue,0).astype(np.float64)
    matrix_stdb = np.std(datablue, 0).astype(np.float64)
    
    matrix_meang = np.mean(datagreen,0).astype(np.float64)
    matrix_stdg = np.std(datagreen, 0).astype(np.float64)
    
    matrix_meanr = np.mean(datared,0).astype(np.float64)
    matrix_stdr = np.std(datared, 0).astype(np.float64)

    data2 = input[(len(input) // 2):len(input)]
    
    data2b = data2[:,:,:,0]
    data2g = data2[:,:,:,1]
    data2r = data2[:,:,:,2]

    gt_test_blue =  []  # size -> ( len(array_alpha), len(data2), rows, cols )
    for alpha in array_alpha:
        sys.stdout.write("\r>  Computing for alpha blue channel = {:.2f}".format(alpha))
        sys.stdout.flush()
        gt_frames = np.where(np.abs(data2b - matrix_meanb) >= (alpha * (matrix_stdb + 2)), 1, 0)
        gt_test_blue.append(gt_frames)
    print("")
    
    gt_test_green =  []  # size -> ( len(array_alpha), len(data2), rows, cols )
    for alpha in array_alpha:
        sys.stdout.write("\r>  Computing for alpha green channel = {:.2f}".format(alpha))
        sys.stdout.flush()
        gt_frames = np.where(np.abs(data2g - matrix_meang) >= (alpha * (matrix_stdg + 2)), 1, 0)
        gt_test_green.append(gt_frames)
    print("")
    
    gt_test_red =  []  # size -> ( len(array_alpha), len(data2), rows, cols )
    for alpha in array_alpha:
        sys.stdout.write("\r>  Computing for alpha red channel = {:.2f}".format(alpha))
        sys.stdout.flush()
        gt_frames = np.where(np.abs(data2r - matrix_meanr) >= (alpha * (matrix_stdr + 2)), 1, 0)
        gt_test_red.append(gt_frames)
    print("")

    gt_test_blue = np.array(gt_test_blue)
    gt_test_green = np.array(gt_test_green)
    gt_test_red = np.array(gt_test_red)
    gt_test = np.where(gt_test_blue & gt_test_green & gt_test_red, 1, 0)
    print("\n ----------------------------------------- \n")

    return matrix_meanr, matrix_stdr, np.array(gt_test)
    
def GaussianColorHSV(input, array_alpha):

    print("\n ---------- Gaussian HSV Method ---------- \n")
    # Training part, for training we only will use the 50% of the data
    data = input[0:len(input) // 2]

    datah = data[:,:,:,0]
    datas = data[:,:,:,1]

    matrix_meanh = np.mean(datah,0).astype(np.float64)
    matrix_stdh = np.std(datah, 0).astype(np.float64)
    
    matrix_means = np.mean(datas,0).astype(np.float64)
    matrix_stds = np.std(datas, 0).astype(np.float64)

    data2 = input[(len(input) // 2):]
    
    data2h = data2[:,:,:,0]
    data2s = data2[:,:,:,1]

    gt_test_h =  []  # size -> ( len(array_alpha), len(data2), rows, cols )
    for alpha in array_alpha:
        sys.stdout.write("\r>  Computing for alpha hue channel = {:.2f}".format(alpha))
        sys.stdout.flush()
        gt_frames = np.where(np.abs(data2h - matrix_meanh) >= (alpha * (matrix_stdh + 2)), 1, 0)
        gt_test_h.append(gt_frames)
    print("")
    
    gt_test_s =  []  # size -> ( len(array_alpha), len(data2), rows, cols )
    for alpha in array_alpha:
        sys.stdout.write("\r>  Computing for alpha saturation channel = {:.2f}".format(alpha))
        sys.stdout.flush()
        gt_frames = np.where(np.abs(data2s - matrix_means) >= (alpha * (matrix_stds + 2)), 1, 0)
        gt_test_s.append(gt_frames)
    print("")

    gt_test_h = np.array(gt_test_h)
    gt_test_s = np.array(gt_test_s)
    gt_test = np.where(gt_test_h & gt_test_s, 1, 0)
    # gt_test = np.where(gt_test1+gt_test2 == 2,1,0)
    print("\n ----------------------------------------- \n")

    return matrix_meanh, matrix_stdh, np.array(gt_test)