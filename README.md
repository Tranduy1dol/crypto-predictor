# make-prediction

---
## Introduction

---
A CLI tool for making predictions using a pre-trained model. This tool is designed to be simple and efficient, allowing users to quickly generate predictions without needing to modify the underlying code.

## Motivation

---
This project is part of my learning journey to explore suitable models for predicting cryptocurrency prices. It also provides midterm reporting results for my Data Mining course.

## Prerequisites

---
- Python 3.x
- Any necessary dependencies. See the `requirements.txt` file for a list of required packages.

### Installing TA-Lib

---
TA-Lib requires special installation steps:

#### On Windows:

---
Download and install the wheel from [unofficial Windows binaries](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)

#### On macOS:

---
```bash
brew install ta-lib
pip install ta-lib
```

#### On Linux:

---
```bash
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
+make
sudo make install
pip install ta-lib
```
# Installation
```bash
pip install -e .
```

## Usage

---
```
Make Prediction CLI
====================

Available commands:
- help     : Show this help message.
- predict  : Run prediction using trained model.
- new      : Initialize a new project setup.
```

First, create your `config.ini` and fill in the required fields. You can use the provided `config.ini.example` as a template. Then use the `new` command to create a new model with your configuration file.
```
predictor new config.ini
```
After trained the model, you can use the `predict` command to make predictions.
```
predictor predict config.ini
```

## Reference

---
For more details, refer to the paper:
- [A Novel Cryptocurrency Price Prediction Model Using GRU, LSTM and bi-LSTM Machine Learning Algorithms](https://www.mdpi.com/2673-2688/2/4/30)

`