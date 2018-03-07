# Week 2

## Delivery

This Week2.md help us to prepare the environment in order to be able to code correctly for each second delivery. We explain the use of each python file and its respective function.

## Main

From main.py it can be executed all the code provided for us.

## Configuration

The configur ation of the database used is in the config.py file. You can specify which dataset you want to use and the names of the root folder and their respective subfolders (Ground Truth and Input images).

## Database

The way how we read and load the files of the dataset. 

## Task 1

We created the function OneSingleGaussian() inside the python file OneSingleGaussian.py. We obtain the mean and std matrices and for each alpha specified in an array of alphas, we segment the foreground and we obtain different segmentations that are evaluated with F1-score. 

## Task 2

We created the function OneSingleGaussianAdapt.py. In the training part we obtain the mean and std matrices, then we apply the adaptative model that depends of two arrays of parameters rho and alpha. The goal is to do the adaptive model and improve the results obtained in the non-adaptive gaussian model.  

## Task 3

## Task 4
For this task, we created GaussianColor.py, that have inside GaussianColorRGB() and GaussianColorHSV(), the first function is a foreground/background segmentation based on RGB. If the value of the pixel fall in the background region in the 3 components, this pixel will be background, otherwise, this pixel will be foreground. In GaussianColorHSV() the aproach is based on use Hue and Saturation channels. If the value of the pixel fall in the background region in both components, this pixel will be background. otherwise, this pixel will be foreground.


## Performance

For each Task, we calculate the True Postive, False Positive, True Negative and False Negative values for each case.

## Metrics

For each Task, we also calculate the precision, recall and f1-score in order to obtain the requested metrics.


