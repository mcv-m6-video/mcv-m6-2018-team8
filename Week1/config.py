import os
# DATABASE = "kitti"
DATABASE = "changedetection"

dir_databases = "databases"
dir_results = 'results'
dir_input = 'input'
dir_gt = 'groundtruth'

if DATABASE == "kitti":
    abs_dir_result = os.path.join(dir_databases, DATABASE, dir_results)
elif DATABASE == "changedetection":
    abs_dir_result = os.path.join(dir_databases, DATABASE, "highway", dir_results)
    abs_dir_gt = os.path.join(dir_databases, DATABASE, "highway", dir_gt)
    abs_dir_input = os.path.join(dir_databases, DATABASE, "highway", dir_input)
else:
    raise ValueError(DATABASE + " DB does not exist!")



