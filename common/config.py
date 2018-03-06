import os

"""
Configuration File:
"""

# ---------------- PARAMETERS ---------------- #
# Change the DATABASE depending on the dataset you want to use
# DATABASE = "kitti"
# DATABASE = "changedetection"
DATABASE = "fall"
# DATABASE = "traffic"
dir_databases = "../databases"
dir_results = 'results'
dir_input = 'input'
dir_gt = 'groundtruth'
# -------------------------------------------- #

if DATABASE == "kitti":
    start_frame = 0
    end_frame = -1
    abs_dir_result = os.path.join(dir_databases, DATABASE, dir_results)
    abs_dir_gt = os.path.join(dir_databases, DATABASE, "flow_noc")
    abs_dir_input = os.path.join(dir_databases, DATABASE, dir_results)

elif DATABASE == "changedetection":
    start_frame = 1050
    end_frame = 1350
    abs_dir_result = os.path.join(dir_databases, DATABASE, "highway", dir_results)
    abs_dir_gt = os.path.join(dir_databases, DATABASE, "highway", dir_gt)
    abs_dir_input = os.path.join(dir_databases, DATABASE, "highway", dir_input)

elif DATABASE == "fall":
    #abs_dir_result = os.path.join(dir_databases, DATABASE, dir_results)
    start_frame = 1460
    end_frame = 1560
    abs_dir_gt = os.path.join(dir_databases, DATABASE, dir_gt)
    abs_dir_input = os.path.join(dir_databases, DATABASE, dir_input)

elif DATABASE == "traffic":
    start_frame = 950
    end_frame = 1050
    #abs_dir_result = os.path.join(dir_databases, DATABASE, dir_results)
    abs_dir_gt = os.path.join(dir_databases, DATABASE,  dir_gt)
    abs_dir_input = os.path.join(dir_databases, DATABASE, dir_input)

else:
    raise ValueError(DATABASE + " DB does not exist!")