Here’s the revised version of your documentation with grammatical and stylistic improvements:

---

# Code Structure Documentation

This document outlines the codebase structure for the **python-trading-app** project.

## Project Overview

The project is designed to automate cryptocurrency trading by training and selecting AI models based on Binance data for
predictions. It includes functionalities for data retrieval, model training, real-time trading, and user interaction
through an enhanced UI.

---  

## Main Components

### 1. **Main Entry Point**

- **File:** `main.py`
- **Purpose:**
    - Starts the application.
    - Handles initial user interactions and login.
    - Initiates the real-time trading process after receiving user input (e.g., trading token).

---  

### 2. **AI Model Training**

- **Folder:** `research`
- **Responsibilities:**
    - Fetches historical cryptocurrency data from Binance.
    - Processes the data into a format suitable for AI modeling.
    - Trains multiple AI models for prediction and selects the most accurate one.
- **Details:**
    - **File:** `gathering_data.ipynb`
        - Requires MongoDB for gathering Bitcoin trading data from Binance.
        - Splits the data into 4-hour durations and saves it to the database.
    - **File:** `training_model.ipynb`
        - Uses data from MongoDB after running `gathering_data`.
        - Converts the data into a DataFrame with six inputs: `open`, `close`, `high`, `low`, `time`, and `volume`.
        - Calculates additional attributes, reshapes the data, and trains a model for value prediction.
        - Exports the trained model to the `models` folder.
    - **File:** `test_live_trading.ipynb`
        - Uses the trained model to make real-time predictions and displays the results.
    - **File:** `backtest.ipynb`
        - Tests the model's accuracy after training by performing a backtest.

---  

### 3. **Model Storage**

- **Folder:** `models`
- **Purpose:**
    - Stores trained models.

---  

### 4. **Libraries**

- **Folder:** `libs`
- **Purpose:**
    - Handles model calculations and predictions.
- **Details:**
    - **File:** `exchange.py`
        - Contains the following functions:
            - `check_login`: Verifies login credentials on Binance.
            - `take_buy_order` and `take_sell_order`: Execute buy/sell orders for a hard-coded amount in `main.py` (
              modifiable as needed).
            - `check_order`: Checks whether an order is finalized.
    - **File:** `process.py`
        - Contains the following functions:
            - `gather_data`: Collects historical cryptocurrency data, including prices, volumes, and other trading
              metrics.
            - `transform_data`: Converts raw data into a state representation for AI modeling or trading decisions.

---  

