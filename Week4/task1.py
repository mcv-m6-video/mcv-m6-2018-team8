import sys
sys.path.append('../.')

from common.config import *
from methods.BlockMatchingOF import *
from common.Database import *
from methods.GaussianMethods import *
from common.extractPerformance import *
from common.metrics import *

if __name__ == "__main__":
    gt_db = Database(abs_dir_gt, start_frame=start_frame, end_frame=end_frame)
    input_db = Database(abs_dir_input, start_frame=start_frame, end_frame=end_frame)

    gt = gt_db.loadDB(im_color=cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
    input = input_db.loadDB(im_color=True)

    i_test = 0

    window_sizes = np.arange(7, 74, 8)
    shift_sizes = np.arange(7, 74, 8)

    np.save("window_sizes.npy", window_sizes)
    np.save("shift_sizes.npy", shift_sizes)

    x_directions = []
    y_directions = []
    motion_images = []

    for i in range(0, len(input), 2):

        cv2.imshow("Previous Frame", input[i])
        cv2.imshow("Next Frame", input[i + 1])
        cv2.waitKey(1)

        total_id = 0
        for idx, ws in enumerate(window_sizes):
            for idy, sh in enumerate(shift_sizes):

                print("{}/{}".format(total_id+1, len(window_sizes) * len(shift_sizes)))

                x, y, y_motion_pred = BlockMatchingOpticalFlow(input[i], input[i+1], window_size=ws, stride=1, shift=sh, im_show_debug=False)
                total_id += 1

                x_directions.append(x)
                y_directions.append(y)
                motion_images.append(y_motion_pred)


        # output_gt_0, output_pred_0, E_0, pepn_0 = MSEN_PEPN(y_gt=gt[i_test], y_pred=y_motion_pred,
        #                                                     of_from_dataset=False, show_error=True)
        # i_test += 1

        np.save("x_directions.npy", x_directions)
        np.save("y_directions.npy", y_directions)
        np.save("motion_images.npy", motion_images)

        # x = np.load("x_directions.npy")
        # y = np.load("y_directions.npy")
        # y_motion_pred = np.load("motion_images.npy")

        # quiver_flow_field(x, y)



