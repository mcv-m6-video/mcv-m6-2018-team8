import os
import cv2

"""
Configuration File:
"""

# ---------------- PARAMETERS ---------------- #
# Change the DATABASE depending on the dataset you want to use
# DATABASE = "kitti"
DATABASE = "highway"
# DATABASE = "fall"
# DATABASE = "traffic"

dir_databases = "../databases"
dir_results = 'results'
dir_input = 'input'
dir_gt = 'groundtruth'

GAUSSIAN_METHOD = 'adaptative'
MORPH_EX = 'opening'

# choose the desire strucutre for the morphological operation (cv2 type)
MORPH_STRUCTURE = cv2.MORPH_RECT
# MORPH_STRUCTURE = cv2.MORPH_CROSS
# MORPH_STRUCTURE = cv2.MORPH_ELLIPSE

# -------------------------------------------- #

if DATABASE == "kitti":
    start_frame = 0
    end_frame = -1
    abs_dir_result = os.path.join(dir_databases, DATABASE, dir_results)
    abs_dir_gt = os.path.join(dir_databases, DATABASE, "flow_noc")
    abs_dir_input = os.path.join(dir_databases, DATABASE, dir_results)

elif DATABASE == "highway":
    start_frame = 1050
    end_frame = 1350
    abs_dir_result = os.path.join(dir_databases, "changedetection", DATABASE, dir_results)
    abs_dir_gt = os.path.join(dir_databases, "changedetection", DATABASE, dir_gt)
    abs_dir_input = os.path.join(dir_databases, "changedetection", DATABASE, dir_input)

elif DATABASE == "fall":
    start_frame = 1460
    end_frame = 1560
    abs_dir_result = None
    abs_dir_gt = os.path.join(dir_databases, DATABASE, dir_gt)
    abs_dir_input = os.path.join(dir_databases, DATABASE, dir_input)

elif DATABASE == "traffic":
    start_frame = 950
    end_frame = 1050
    abs_dir_result = None
    abs_dir_gt = os.path.join(dir_databases, DATABASE,  dir_gt)
    abs_dir_input = os.path.join(dir_databases, DATABASE, dir_input)

else:
    raise ValueError(DATABASE + " DB does not exist!")