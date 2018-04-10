
# Week 5

This Week's README help us to prepare the environment in order to be able to code correctly for each fifth delivery. We explain the use of each python file and its respective function.

## Description by Tasks

### Task 1
We have created the python file `Tracking_KalmanFilter()` and `Tracking_CamShift()` in order to track the vehicles of Highway and Traffic sequences with Kalman Filter and CamShift, respectively. Before we have subtracted the background with the adaptive gaussian model (`OneSingleGaussianAdapt()` from Week2). We have chosen the best rho value and alphas for each dataset from Week3. Then, we have post-processed the results applying hole filling and opening to improve the results obtained in the Adaptive Gaussian model.

### Task 2
In this part, we have implemented a function `SpeedDetector()` in `Tracking_KalmanFilter()` and `Tracking_CamShift()` that estimates the speed of the cars.

### Task 3
In this task, we have recorded our own video in the UAB and we have done the same pipeline carried out during both tasks before.


## Execution usage
### Vehicle Tracker (Task 1)
Execute the `Week5.py` to execute the `OneSingleGaussianAdapt()` from `GaussianMethods.py`, to apply Hole Filling from the function `Holefilling()` and Opening from `MorphologicalTransformation()` and finally `Tracking_KalmanFilter()` and `Tracking_CamShift()` for tracking vehicles with the respective method, Kalman Filter (Task 1.1) and CamShift (Task 1.2).

```sh
$ python Week5.py
```

### Speed Estimator (Task 2)
The Speed Estimator is done in `Tracking_KalmanFilter()` and `Tracking_CamShift()`. To get it, we have to execute `Week5.py` too. 

```sh
$ python Week5.py
```

### Our own study (Task 3)
Execute the `Week5ownvideo.py` to execute our own video with the same pipeline of the tasks before

```sh
$ python Week5ownvideo.py
```
