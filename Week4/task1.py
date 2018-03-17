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

    gt = gt_db.loadDB(im_color=False)
    input = input_db.loadDB(im_color=True)

    for i in range(0, len(input), 2):
        x, y = BlockMatchingOpticalFlow(input[i], input[i+1])
        quiver_flow_field(x, y)