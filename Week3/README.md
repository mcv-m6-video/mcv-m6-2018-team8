# Week 3

This Week's README help us to prepare the environment in order to be able to code correctly for each second delivery. We explain the use of each python file and its respective function.

## Description by Tasks

### Task 1
We have created the python file `Holefilling.py` in order to post-process the results of the best configuration in Week 2 that was the adaptive gaussian model (`ÒneSingleGaussianAdapt()` from Week2). We have chosen the best rho value and swept an array of alphas (10 points from 0 to 5). The goal is to post-process it with hole filling and improve the results obtained in the adaptive gaussian model of the week before with each dataset.

We have used the function `binary_fill_holes()` of the library scipy.ndimage.morphology. 

### Task 2
In this approach, we try to reduce the noise in order to improve results by filtering small regions based on their area.

We have used the function `remove_small_objects()` of the library skimage.morphology.

### Task 3
In this task, we have explored with other morphological filters and combinations to improve AUC for foreground pixels. We have implemented different Morphological Transformations as Erosion, Dilation, Opening, Closing, Gradient, Top-Hat, Black-Hat.

We have imported some functions from OpenCV, `cv2.erode()`, `cv2.dilate()` and `cv2.morphologyEx()`.


### Task 4

### Task 5


## Execution usage
### Holefilling (Task 1)




