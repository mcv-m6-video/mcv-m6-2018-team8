# Video Surveillance for Road Traffic Monitoring

## Introduction 

The goal of this project is to learn the basic concepts and techniques related to video sequences processing, mainly for surveillance applications. The main techniques of video processing will be applied in the context of video surveillance: moving object segmentation, motion estimation and compensation and video object tracking are basic components of many video processing systems. The performance of the developed techniques will be measured using standard metrics for video analysis.

## Authors

Team 8:
- _A. Casadevall, arnau.casadevall.saiz@gmail.com - [acasadevall](https://github.com/acasadevall)_
- _A. Salvador, adrisalva122@gmail.com - [AdrianSalvador](https://github.com/AdrianSalvador)_
- _T. Pérez, tonipereztorres@gmail.com - [tpereztorres](https://github.com/tpereztorres)_

## Project Schedule

- Week 1: Introduction to video sequence analysis and evaluation
- Week 2: Background estimation (not done yet!)
- Week 3: Foreground estimation (not done yet!)
- Week 4: Video stabilization (not done yet!)
- Week 5: Region tracking (not done yet!)

## Common Files and Metrics

### Peformance

For each Task, we calculate the True Postive, False Positive, True Negative and False Negative values for each case.

The Performance has two methods, depending on the number of bnarried parameters you want:

- None or 1 sweep parameter: `extractPerformance()`
- Two sweep parameters: `extractPerformance_2Params()`

### Metrics

For each Task, we also calculate the precision, recall and f1-score in order to obtain the requested metrics.

Methods for metrics:
- None or 1 sweep parameter: `metrics()`
- Two sweep parameters: `metrics_2Params()`
- AUC score: `getAUC()`
- Plot Precision-Recall: `plotPrecisionRecall()`
- Plot F1-Score (2D): `plotF1Score2D()`
- Plot F1-Score (3D): `plotF1Score3D()`

## Configuration

The configuration file is located in `common/config.py`. This file will be upgraded for the new features. You can specify which dataset you want to use, select the names of the root folder and their respective subfolders (Ground Truth and Input images) and other parameters.

## Database

All the database are located in the `database/`. Choose which want to use in the `config.py` file by selection the desired `DATABASE` global variable (view in the file)

The database availables are:
- Changedetection (Highway)
- Fall
- Traffic
- Kitty (for the Optical Flow apporach)
