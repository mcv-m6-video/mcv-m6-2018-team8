# Week 1

This Week's README help us to prepare the environment in order to be able to code correctly for the very first delivery. We explain the use of each python file and its respective function.

Main goals for this Week:

- Understand and become familiar with the programming framework used in the project (Python).
- Learn about the databases to be used
- Implement the evaluation metrics and graphs used during the module.
- Read / write sequences of images and associated segmentation ground truth.

## Description by Tasks

### Task 1

Segmentation metrics. Understand precision and recall.

### Task 2

Segmentation metrics. Temporal analysis.

### Task 3

Optical flow evaluation metrics.

### Task 4

De-synchornized results.

### Task 5

Visual representation optical flow.

## Execution usage

For this Week we have only implement one `main.py` where depending on the DATASET (if is for Optical Flow or not), the code will be execute a diferent performance analysis.

```sh
$ python main.py
```

### Non-Optical Flow
Execute the `main.py` to execute the `metrics.py`, `temporalAnalysis()`, `temporalAnalysis()` and `FramePerformancedelay.py`. These functions will extract all the metrics about the DATASET (Precision, Recall, F1-score) and other information about the temporal analysis.

### Optical Flow
Execute the `main.py` to execute the `metrics.py` to extract the Optical Flow information: `MSEN_PEP()`, `plotHistogram()` and `ShowOpticalFlow()`.
