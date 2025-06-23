# Neurojourney for Python

[![Platform](https://img.shields.io/badge/platform-desktop-orange.svg)](https://github.com/neurojourney/neurojourney-mkyk-python)
[![Languages](https://img.shields.io/badge/language-python-orange.svg)](https://github.com/neurojourney/neurojourney-mkyk-python)
[![Commercial License](https://img.shields.io/badge/license-Commercial-brightgreen.svg)](https://github.com/neurojourney/neurojourney-mkyk-python/blob/master/LICENSE.md)

## Introduction
This project is to MKYK solutions for emotions on heart. 

<br/>

## Requirements
The requirements for this project are:
- Python >= 3.9.x
- contourpy==1.3.2
- cycler==0.12.1
- fonttools==4.58.0
- joblib==1.5.1
- kiwisolver==1.4.8
- matplotlib==3.10.3
- numpy==2.2.5
- packaging==25.0
- pandas==2.2.3
- pillow==11.2.1
- pyparsing==3.2.3
- python-dateutil==2.9.0.post0
- pytz==2025.2
- scikit-learn==1.7.0
- scipy==1.15.3
- seaborn==0.13.2
- six==1.17.0
- threadpoolctl==3.6.0
- tqdm==4.67.1
- tzdata==2025.2
- xgboost==3.0.2

<br/>

## Installation

### Step 1: Clone this repository and move directory
You can **clone** the project from the [repository](https://github.com/neurojourney/neurojourney-mkyk-python).

```
// Clone this repository
git clone Git@github.com:neurojourney/neurojourney-mkyk-python.git

// Move to the repository
cd neurojourney-mkyk-python
```

### Step 2: Install third-party libraries
Please install third-party libraries using [requirements.txt](https://github.com/emotionist/neurojourney-mkyk-python/tree/master/requirements.txt).

```
pip install -r requirements.txt
```

## Run APp

### Step 1: Install Neurojourney project and third-party libraries for you custom app.
You can install neurojourney project and third-party libraries as described in the [README.md](https://github.com/neurojourney/neurojourney-mktk-python/blob/master/README.md)

### Step 2: Run project
You can run application by executing python. 

```
// Run the app
python test_and_train.py
```

### Step 3: Get model and results
You can find the trained model file focus_model.pkl in the current directory.
The evaluation results will be printed to the terminal.

For example:
📊 Train Results:
  R² Score      : 0.8991
  MAE           : 0.0591
  RMSE          : 0.0082
  Correlation   : 0.9538
  Accuracy      : 0.8725
📊 Test Results:
  R² Score      : 0.7353
  MAE           : 0.0920
  RMSE          : 0.0213
  Correlation   : 0.8588
  Accuracy      : 0.8370
