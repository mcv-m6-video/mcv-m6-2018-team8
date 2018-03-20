import sys
sys.path.append('../.')

from common.config import *
from methods.BlockMatchingOF import *
from common.Database import *
from methods.GaussianMethods import *
from common.extractPerformance import *
from common.metrics import *
from time import time
from methods.Farneback import *
from methods.LucasKanade import *

if __name__ == "__main__":
    gt_db = Database(abs_dir_gt, start_frame=start_frame, end_frame=end_frame)
    input_db = Database(abs_dir_input, start_frame=start_frame, end_frame=end_frame)

    gt = gt_db.loadDB(im_color=cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
    input = input_db.loadDB(im_color=True)

    dir_to_save = "output_{}".format(OF_TYPE)
    if not os.path.exists(dir_to_save):
        raise ValueError("{} does not exist!".format(dir_to_save))

    # window_sizes = np.load(os.path.join(dir_to_save, "window_sizes_{}.npy".format(OF_TYPE)))
    # shift_sizes = np.load(os.path.join(dir_to_save, "shift_sizes_{}.npy".format(OF_TYPE)))

    window_sizes = np.array([7, 15, 23, 31, 39])
    shift_sizes = np.array([7, 15, 23, 31, 39, 47, 55, 63, 71])

    stride = 10
    i_test = 0
    for i in range(0, len(input), 2):

        total_id = 0

        pepn_error = []
        msen_error = []

        for idx, ws in enumerate(window_sizes):
            for idy, ss in enumerate(shift_sizes):

                start = time()
                print("{}/{} (image {})".format(total_id + 1, len(window_sizes) * len(shift_sizes), i_test))
                print("window size {} - shift size {}".format(ws, ss))

                if OF_TYPE == "BlockMatching":
                    x_direction = np.load(
                        os.path.join(dir_to_save, "x_directions_{}_{}_s{}_{}_{}.npy".format(ws, ss, stride, OF_TYPE, i_test)))
                    y_direction = np.load(
                        os.path.join(dir_to_save, "y_directions_{}_{}_s{}_{}_{}.npy".format(ws, ss, stride, OF_TYPE, i_test)))
                    y_motion_pred = np.load(
                        os.path.join(dir_to_save, "motion_images_{}_{}_s{}_{}_{}.npy".format(ws, ss, stride, OF_TYPE, i_test)))

                    (nr, nc) = gt[i_test].shape[:2] - ws
                    y_motion_pred = cv2.resize(y_motion_pred, (nc, nr), interpolation=cv2.INTER_CUBIC)

                else:
                    y_motion_pred = np.load(
                        os.path.join(dir_to_save, "motion_images_{}_{}_s{}_{}_{}.npy".format(ws, ss, stride, OF_TYPE, i_test)))

                print("----------------------------")

                total_id += 1


                output_gt_0, output_pred_0, E_0, pepn_0 = MSEN_PEPN(y_gt=gt[i_test], y_pred=y_motion_pred,
                                                                    of_from_dataset=False, show_error=False)

                pepn_error.append(pepn_0)
                msen_error.append(np.mean(E_0))

        print("PEPN Min: {}".format(np.min(np.array(pepn_error))))
        print("MSEN Min: {}".format(np.min(np.array(msen_error))))


        if OF_TYPE == "BlockMatching":
            X, Y = np.meshgrid(window_sizes, shift_sizes, indexing='ij')
            Z = (np.array(pepn_error)*100).reshape(X.shape)

            plt.figure()
            ax = plt.axes(projection='3d')
            surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
            ax.contour3D(X, Y, Z, 50, cmap='viridis')
            ax.set_xlabel("Window size")
            ax.set_ylabel("Shift size")
            ax.set_zlabel("PEPN Error %")
            plt.colorbar(surf)
            plt.savefig(os.path.join(dir_to_save, "plot_3d_pepn.png"), bbox_inches='tight', frameon=False)
            plt.show()

            Z = np.array(msen_error).reshape(X.shape)

            plt.figure()
            ax = plt.axes(projection='3d')
            surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
            ax.contour3D(X, Y, Z, 50, cmap='viridis')
            ax.set_xlabel("Window size")
            ax.set_ylabel("Shift size")
            ax.set_zlabel("MESN Error")
            plt.colorbar(surf)
            plt.savefig(os.path.join(dir_to_save, "plot_3d_msen.png"), bbox_inches='tight', frameon=False)
            plt.show()

        else:
            plot2D(x_axis=window_sizes, y_axis=np.array(pepn_error)*100,
                   x_label='Window size', y_label='PEPN Error %', legend_label="{} - {}".format(DATABASE, OF_TYPE))
            plot2D(x_axis=window_sizes, y_axis=np.array(msen_error),
                   x_label='Window size', y_label='MESN Error', legend_label="{} - {}".format(DATABASE, OF_TYPE))

        i_test += 1

        # quiver_flow_field(x, y)