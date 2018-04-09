import cv2
import numpy as np
import warnings
import sys, os

def extractPerformance(gt, gt_test, array_params=None):

    if isinstance(gt, (list,)):
        gt = np.array(gt)

    if isinstance(gt_test, (list,)):
        gt_test = np.array(gt_test)

    print("Performance ...")

    gt = np.where((gt <= 50), 0, gt)
    gt = np.where((gt == 255), 1, gt)
    gt = np.where((gt == 85), -1, gt)
    gt = np.where((gt == 170), -1, gt)

    if not np.array_equal(np.unique(gt_test), [0, 1]):
        gt_test[gt_test!=0] = 1

    TP_list = []
    FP_list = []
    TN_list = []
    FN_list = []

    dim = 1
    if np.array(gt_test).ndim > 3:
        dim = len(gt_test)

    for a in range(dim):
        # True Positive (TP): # pixels correctly segmented as foreground
        TP = np.sum(np.logical_and(gt_test == 1, gt == 1)) if dim == 1 else np.sum(np.logical_and(gt_test[a] == 1, gt == 1))

        # True Negative (TN): pixels correctly detected as background
        TN = np.sum(np.logical_and(gt_test == 0, gt == 0)) if dim == 1 else np.sum(np.logical_and(gt_test[a] == 0, gt == 0))

        # False Positive (FP): pixels falsely segmented as foreground
        FP = np.sum(np.logical_and(gt_test == 1, gt == 0)) if dim == 1 else np.sum(np.logical_and(gt_test[a] == 1, gt == 0))

        # False Negative (FN): pixels falsely detected as background
        FN = np.sum(np.logical_and(gt_test == 0, gt == 1)) if dim == 1 else np.sum(np.logical_and(gt_test[a] == 0, gt == 1))

        TP_list.append(TP)
        FP_list.append(FP)
        TN_list.append(TN)
        FN_list.append(FN)

        sys.stdout.write("\r>  Computing ... {:.2f}%".format((a+1) * 100 / dim))
        sys.stdout.flush()

    print("\n\nSummary: TP, FP, TN, FN")
    print("------------------------------------------------------------")
    if array_params is None:
        print("TP: {}\tFP: {}\tTN: {}\tFN: {}".format(TP_list[0], FP_list[0], TN_list[0], FN_list[0]))
    else:
        for id, param in enumerate(array_params):
            print("For param = {:.4f} - TP: {} \tFP: {} \tTN: {} \tFN: {}".format(param, TP_list[id], FP_list[id], TN_list[id], FN_list[id]))
    print("------------------------------------------------------------")

    return TP_list, FP_list, TN_list, FN_list

def extractPerformance_2Params(gt, gt_test, array_params_a, array_params_b, im_show_performance=True):

    if isinstance(gt, (list,)):
        gt = np.array(gt)

    if isinstance(gt_test, (list,)):
        gt_test = np.array(gt_test)

    print("Performance ...")

    gt = np.where((gt <= 50), 0, gt) #Static
    gt = np.where((gt == 255), 1, gt) #Motion
    gt = np.where((gt == 85), -1, gt) #Unknown
    gt = np.where((gt == 170), -1, gt) #Unknown

    if not np.array_equal(np.unique(gt_test), [0, 1]):
        gt_test[gt_test!=0] = 1

    TP_list = []
    FP_list = []
    TN_list = []
    FN_list = []

    total_id = 0
    for a in range(len(array_params_a)):
        for b in range(len(array_params_b)):
            # True Positive (TP): # pixels correctly segmented as foreground
            TP = np.sum(np.logical_and(gt_test[total_id] == 1, gt == 1))

            # True Negative (TN): pixels correctly detected as background
            TN = np.sum(np.logical_and(gt_test[total_id] == 0, gt == 0))

            # False Positive (FP): pixels falsely segmented as foreground
            FP = np.sum(np.logical_and(gt_test[total_id] == 1, gt == 0))

            # False Negative (FN): pixels falsely detected as background
            FN = np.sum(np.logical_and(gt_test[total_id] == 0, gt == 1))

            TP_list.append(TP)
            FP_list.append(FP)
            TN_list.append(TN)
            FN_list.append(FN)

            if im_show_performance:
                compareImages(gt, gt_test[total_id], delay_ms=1000)

            total_id += 1

        sys.stdout.write("\r>  Computing ... {:.2f}%".format((total_id+1)*100 / (len(array_params_a)*len(array_params_b))))
        sys.stdout.flush()

    print("\n\nSummary: TP, FP, TN, FN")
    print("------------------------------------------------------------")
    total_id = 0
    for id_a, param_a in enumerate(array_params_a):
        for id_b, param_b in enumerate(array_params_b):
            print("For params = {:.4f} | {:.4f} - TP: {} \tFP: {} \tTN: {} \tFN: {}".format(param_a, param_b, TP_list[total_id], FP_list[total_id], TN_list[total_id], FN_list[total_id]))
            total_id += 1
    print("------------------------------------------------------------")

    return TP_list, FP_list, TN_list, FN_list

def checkImageForVisualise(im):

    if isinstance(im, (list,)):
        im = np.array(im)

    if im.dtype == 'bool':
        if not np.array_equal(np.unique(im), [0, 1]):
            im[im < 0] = 0

        im = im.astype(np.uint8)

    elif not im.dtype == 'uint8':
        im = im.astype(np.uint8)
        warnings.warn("The {} image it should be uint8 and it may not be displayed correctly.".format(im.dtype))

    return im

def compareImages(input_a, input_b, delay_ms=0):

    input_a = checkImageForVisualise(input_a)
    input_b = checkImageForVisualise(input_b)

    assert(len(input_a) == len(input_b))

    for i in range(len(input_a)):
        # [0,1] to [0,255]
        cv2.imshow("Input A", cv2.convertScaleAbs(input_a[i], alpha=np.iinfo(input_a[i].dtype).max / np.amax(input_a[i])))
        cv2.imshow("Input B", cv2.convertScaleAbs(input_b[i], alpha=np.iinfo(input_b[i].dtype).max / np.amax(input_b[i])))

        cv2.waitKey(delay_ms)

    cv2.destroyAllWindows()

def MakeYourGIF(input, path_to_save='video.gif'):

    if isinstance(input, (list,)):
        input = np.array(input)

    assert(input.ndim <= 4)

    input = checkImageForVisualise(input)

    import imageio
    with imageio.get_writer(path_to_save, mode='I') as writer:

        for i, file in enumerate(input):
            file = cv2.convertScaleAbs(file, alpha=np.iinfo(file.dtype).max / np.amax(file))
            # cv2.imshow("GIF", file)
            # cv2.waitKey(1)

            writer.append_data(file)

            sys.stdout.write("\r>  Making GIF {} ... {:.2f}%".format(os.path.basename(path_to_save), i*100/len(input)))
            sys.stdout.flush()

        print("")

        cv2.destroyAllWindows()