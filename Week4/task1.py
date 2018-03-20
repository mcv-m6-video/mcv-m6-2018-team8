import sys
sys.path.append('../.')

from common.config import *
from methods.BlockMatchingOF import *
from common.Database import *
from methods.GaussianMethods import *
from common.extractPerformance import *
from common.metrics import *
from methods.Farneback import *
from methods.LucasKanade import *
from time import time

if __name__ == "__main__":
    gt_db = Database(abs_dir_gt, start_frame=start_frame, end_frame=end_frame)
    input_db = Database(abs_dir_input, start_frame=start_frame, end_frame=end_frame)

    gt = gt_db.loadDB(im_color=cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
    input = input_db.loadDB(im_color=True)



    dit_to_save = "output_{}".format(OF_TYPE)
    if not os.path.exists(dit_to_save):
        os.mkdir(dit_to_save)

    # window_sizes = np.arange(2, 500, 10)  # up to 150 because of "somthing strange in Lucas Kanade
    # shift_sizes = np.arange(1, 2, 1)

    window_sizes = [ 7, 15, 23, 31, 39, 47, 55, 63, 71]
    shift_sizes =[ 7, 15, 23, 31, 39, 47, 55, 63, 71]

    np.save(os.path.join(dit_to_save,"window_sizes_{}.npy".format(OF_TYPE)), window_sizes)
    np.save(os.path.join(dit_to_save,"shift_sizes_{}.npy".format(OF_TYPE)), shift_sizes)

    stride = 10
    i_test = 1 # start with frame 1
    for i in range(2, len(input), 2):

        x_directions = []
        y_directions = []
        motion_images = []

        # cv2.imshow("Previous Frame", input[i])
        # cv2.imshow("Next Frame", input[i + 1])
        # cv2.waitKey(1)

        total_id = 0
        for idx, ws in enumerate(window_sizes):
            for idy, ss in enumerate(shift_sizes):

                start = time()
                print("{}/{} (image {})".format(total_id+1, len(window_sizes) * len(shift_sizes), i_test))
                print("window size {} - shift size {}".format(ws, ss))

                if OF_TYPE == "LucasKanade":
                    y_motion_pred = lucas_kanade_np(input[i], input[i + 1], win=ws)
                elif OF_TYPE == "Farneback":
                    y_motion_pred = Farneback(input[i], input[i+1], win=ws)
                elif OF_TYPE == "BlockMatching":
                    x, y, y_motion_pred = BlockMatchingOpticalFlow(input[i], input[i+1], window_size=ws, stride=stride, shift=ss, im_show_debug=False)
                    x_directions.append(x)
                    y_directions.append(y)
                    np.save(os.path.join(dit_to_save,
                                         "x_directions_{}_{}_s{}_{}_{}.npy".format(ws, ss, stride, OF_TYPE, i_test)),x)
                    np.save(os.path.join(dit_to_save,
                                         "y_directions_{}_{}_s{}_{}_{}.npy".format(ws, ss, stride, OF_TYPE, i_test)),y)
                else:
                    raise ValueError("{} does not exist!".format(OF_TYPE))

                print("Tima Elapsed: {}s".format(time() - start))

                print("----------------------------")

                total_id += 1

                motion_images.append(y_motion_pred)

                np.save(os.path.join(dit_to_save,"motion_images_{}_{}_s{}_{}_{}.npy".format(ws, ss, stride, OF_TYPE, i_test)), y_motion_pred)

                output_gt_0, output_pred_0, E_0, pepn_0 = MSEN_PEPN(y_gt=gt[i_test], y_pred=y_motion_pred,
                                                                    of_from_dataset=False, show_error=False)

        i_test += 1

        np.save(os.path.join(dit_to_save,"x_all_directions_{}_{}.npy".format(OF_TYPE, i)), x_directions)
        np.save(os.path.join(dit_to_save,"y_all_directions_{}_{}.npy".format(OF_TYPE, i)), y_directions)
        np.save(os.path.join(dit_to_save,"all_motion_images_{}_{}.npy".format(OF_TYPE, i)), motion_images)

        # x = np.load("x_directions.npy")
        # y = np.load("y_directions.npy")
        # y_motion_pred = np.load("motion_images.npy")

        # quiver_flow_field(x, y)



