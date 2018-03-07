# Week 2

This Week's README help us to prepare the environment in order to be able to code correctly for each second delivery. We explain the use of each python file and its respective function.

## Description by Tasks

### Task 1
We have created the function `OneSingleGaussian()` inside the python file `OneSingleGaussian.py`. We obtain the mean and std matrices for each alpha specified in an array of alphas, then we segment the foreground and we obtain different segmentations that are evaluated with F1-score.

### Task 2
We have created the function `OneSingleGaussianAdapt()`. In the training part we obtain the mean and std matrices, then we apply the adaptative model that depends of two arrays of parameters rho and alpha. The goal is to do the adaptive model and improve the results obtained in the non-adaptive gaussian model.

### Task 3
The idea of this task is to compare several background susbtraction approaches of the state-of-the-art and compare them with our results. See in the compare-methods section to view which methods have been used.

### Task 4
For this task, we created `GaussianColor.py`, that have inside `GaussianColorRGB()` and `GaussianColorHSV()`, the first function is a foreground/background segmentation based on RGB. If the value of the pixel fall in the background region in the 3 components, this pixel will be background, otherwise, this pixel will be foreground. In `GaussianColorHSV()` the approach is based on use Hue and Saturation channels. If the value of the pixel fall in the background region in both components, this pixel will be background. otherwise, this pixel will be foreground.

## Execution usage
#### OneSingleGaussian and OneSingleGaussianAdapt (Task 1.x - 2.x)
Execute the main.py to execute the `OneSingleGaussian` or `OneSingleGaussianAdapt` and extract barried parameters by using `extractPerformance` and `extractPerformance_2Params` function, respectively.
These functions are expected to have the input input images with the array of params (eg. nunmpy.linspace(0,5,10))

By now you have to change the functions you want to use by commenting lines.
```sh
$ python main.py
```

#### Comparision of methods of the state-of-the-art and OneSingleGaussianAdapt (Task 3)

The methods available to compare are the following:
- MOG (OpenCV 3.1.0 [documentation ](https://docs.opencv.org/3.1.0/d6/da7/classcv_1_1bgsegm_1_1BackgroundSubtractorMOG.html))
- MOG2 (OpenCV 3.1.0 [documentation ](https://docs.opencv.org/3.1.0/d7/d7b/classcv_1_1BackgroundSubtractorMOG2.html))
- KNN (OpenCV 3.1.0 [documentation](https://docs.opencv.org/3.1.0/db/d88/classcv_1_1BackgroundSubtractorKNN.html))
- SuBSENSE + LOBSTER (from Etheron's [Github](https://github.com/ethereon/subsense))

There might be some mehotds that can not work depending on the OpenCV version. If the method does not exist it will raise a warning but the execution will not crash. All the methods from OpenCV have been testes with a OpenCV 3.3.1.

The last method from SuBSENSE + LOBSTER is located in the `thirdparty` folder where there are all the cpp files and a possible `Makefile` and/or `CMakelist.txt` For now, this method only works with a Linux x64_86 (with libSubsense.so)

To build the library (with Linux and OpenCV 2.4.11 tested):
```
$ cd thirdparty/
$ mkdir build
$ cd build
$ cmake ..
$ mv libSubsense.so ../ #the library needs to be at the same folder of the Subsense.py
```

Execute the code:

```sh
$ python comapare-methods.py
```

#### Color-based approaches (Task 4)
RGB color-base: `GaussianColorRGB()`
HSV color-base: `GaussianColorHSV()`

By now you have to change the functions you want to use by commenting lines.

Execute the code:

```sh
$ python color-test.py
```
