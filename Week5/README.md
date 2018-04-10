
# Week 5

This Week's README help us to prepare the environment in order to be able to code correctly for each fourth delivery. We explain the use of each python file and its respective function.

## Description by Tasks

### Task 1
We have created the python file `Holefilling.py` in order to track the vehicles of Highway and Traffic sequences with Kalman Filterost-process the results of the best configuration in Week 2 that was the adaptive gaussian model (`OneSingleGaussianAdapt()` from Week2). We have chosen the best rho value and swept an array of alphas (10 points from 0 to 5). The goal is to post-process it with hole filling and improve the results obtained in the adaptive gaussian model of the week before with each dataset.

We have used the function `binary_fill_holes()` of the library scipy.ndimage.morphology. 

### Task 2
With this approach, we try to reduce the noise in order to improve results by filtering small regions based on their area.

We have used the function `remove_small_objects()` of the library skimage.morphology.

### Task 3
In this task, we have explored with other morphological filters and combinations to improve AUC for foreground pixels. We have implemented different Morphological Transformations as Erosion, Dilation, Opening, Closing, Gradient, Top-Hat, Black-Hat.

We have imported some functions from OpenCV, `cv2.erode()`, `cv2.dilate()` and `cv2.morphologyEx()`.


### Task 4

We have searched for different existing techniques and we have implemented the method "Shadow removal with blob-based morphological reconstruction for error correction" (Xu, Landabaso, Pardàs, ICASSP 2005)

### Task 5

Here it is time to compare the results respect to the baseline from the Week before.

## Execution usage
### Vehicle Tracker (Task 1)
Execute the `Week5.py` to execute the `OneSingleGaussianAdapt()` from `GaussianMethods.py`, to apply Hole Filling from the function `Holefilling()` and Opening from `MorphologicalTransformation()` and finally `Tracking_KalmanFilter()` and `Tracking_CamShift()` for tracking vehicles with the respective method, Kalman Filter (## Task 1.1) and CamShift (## Task 1.2)

```sh
$ python Week5.py
```

### Area Filtering (Task 2)
Execute the `task2.py` to execute `AreaFiltering()` to remove noise. 

```sh
$ python task2.py
```

### Morphological Transformations (Task 3)
Execute the `task3.py` to execute `MorphologicalTransformation()` to apply different morphological filters.

```sh
$ python task3.py
```

### Shadow Removal (Task 4)
Execute the `task4.py` to execute `ShadowRemoval()` to remove the shadows of the foregorund.

```sh
$ python task4.py
```

