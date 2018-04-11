# Video Surveillance for Road Traffic Monitoring

## Introduction 

The goal of this project is to learn the basic concepts and techniques related to video sequences processing, mainly for surveillance applications. The main techniques of video processing will be applied in the context of video surveillance: moving object segmentation, motion estimation and compensation and video object tracking are basic components of many video processing systems. The performance of the developed techniques will be measured using standard metrics for video analysis.

## Authors

Team 8:
- _A. Casadevall, arnau.casadevall.saiz@gmail.com - [acasadevall](https://github.com/acasadevall)_
- _A. Salvador, adrisalva122@gmail.com - [AdrianSalvador](https://github.com/AdrianSalvador)_
- _T. Pérez, tonipereztorres@gmail.com - [tpereztorres](https://github.com/tpereztorres)_

## Project Schedule

- Week 1: [Introduction to video sequence analysis and evaluation](Week1)
- Week 2: [Background estimation](Week2)
- Week 3: [Foreground estimation](Week3)
- Week 4: [Video stabilization](Week4)
- Week 5: [Region tracking](Week5)

## Slides Final Project

You can see the slides of the final project from this _[link](https://docs.google.com/presentation/d/1GgvD_Hnn-2GLvDzik_qr1JNEuMHfOUU9SbkHbf38tfo/edit?usp=sharing)_ 

## Report

You can read the report of the project from this _[link](https://www.overleaf.com/read/fypcgqxsrmtk)_

## Common Files and Metrics

### Peformance

Located in `common/extractPerformance.py`.
Calculate the True Postive, False Positive, True Negative and False Negative values for each case.

The Performance has two methods, depending on the number of the sweep parameters you want:

- None or 1 sweep parameter: `extractPerformance()`
- Two sweep parameters: `extractPerformance_2Params()`
- Make Your GIF: `MakeYourGIF()`
- Show different sets of images: `compareImages()`
- Helper functions for visualise: `checkImageForVisualise()`

### Metrics

Located in `common/metrics.py`.
Include all the functions to to calculate the precision, recall and f1-score in order to obtain the requested metrics.

Methods for metrics:
- None or 1 sweep parameter: `metrics()`
- Two sweep parameters: `metrics_2Params()`
- AUC score: `getAUC()`
- Plot Precision-Recall: `plotPrecisionRecall()`
- Plot F1-Score (2D): `plotF1Score2D()`
- Plot F1-Score (3D): `plotF1Score3D()`

For Optical Flow:
- Mean Square Error and Percentage of Erroneous Pixels in Non-Occluded Areas: `MSEN_PEPN()`
- Read Optical Flow images: `ReadOpticalFlow()`
- Show Optical Flow images: `ShowOpticalFlow()`
- Plot Histogram for MSE: `plotHistogram()`

Others (view description in `common/metrics.py`)
- `findParams()`

### Methods

Some methods are included in their respective Week folder, but other more general are lcoated in `common/methods`.
You can find more information in each method:

- `AreaFiltering.py`
- `BlockMathingOF.py`
- `Farneback.py`
- `GaussianMethods.py`
- `LucasKanade.py`
- `ShadowRemoval.py`

## Configuration

The configuration file is located in `common/config.py`. This file will be upgraded for the new features. You can specify which dataset you want to use, select the names of the root folder and their respective subfolders (Ground Truth and Input images) and other parameters.
More info in `common/config.py`.

## Database

All the database are located in the `databases/`. Choose which want to use in the `config.py` file by selection the desired `DATABASE` global variable (view in the file)

The database availables are:
- Highway (Changedetection)
- Fall
- Traffic
- Kitty (for the Optical Flow approach)
