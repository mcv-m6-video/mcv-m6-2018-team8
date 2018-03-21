from BlockMatchingOF import *
from CompensateImage import *
from extractPerformance import *
import numpy as np
import cv2

if __name__ == "__main__":

    i_test = 0

    window_sizes = np.arange(15, 16, 8)
    shift_sizes = np.arange(15, 16, 8)

    x_directions = []
    y_directions = []
    motion_images = []
    video = "m6_week4_task2.mp4"
    cap = cv2.VideoCapture(video)

    N = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.cv.CV_CAP_PROP_FPS))

    status1, prev = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    for i in range(0, N - 1):
        status2, cur = cap.read()
        cur_gray = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)

        total_id = 0
        for idx, ws in enumerate(window_sizes):
            for idy, sh in enumerate(shift_sizes):
                print("{}/{}".format(total_id + 1, len(window_sizes) * len(shift_sizes)))

                x, y, y_motion_pred = BlockMatchingOpticalFlow(prev_gray, cur_gray, window_size=ws, stride=20, shift=sh, motion_estimation="forward", im_show_debug=False)
                total_id += 1

                (nr, nc) = cur.shape[:2] - ws
                y_motion_pred = cv2.resize(y_motion_pred, (nc, nr), interpolation=cv2.INTER_CUBIC)

                x_directions.append(x)
                y_directions.append(y)
                motion_images.append(y_motion_pred)

        prev_gray = cur_gray

        i_test += 1

    block_size = 7
    imagescomp = []

    for i in range(0, N - 1):
        fr = cap.read()
        fr_gray = cv2.cvtColor(fr[1], cv2.COLOR_BGR2GRAY)
        frame = np.array(fr_gray)
        x_blocks = int(frame.shape[0] // block_size)
        y_blocks = int(frame.shape[1] // block_size)
        comp_img, u, v = CompensateImage(frame, np.array(x_directions[i]), np.array(y_directions[i]), block_size, x_blocks, y_blocks)
        comp_img = np.array(comp_img)
        imagescomp.append(comp_img)

    final_images = np.array(imagescomp)

    MakeYourGIF(final_images, "video.gif")