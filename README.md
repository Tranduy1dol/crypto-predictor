# python-trading-app

## Motivation

This project is for the final term exam in the Japanese Technology Topic Subject.

---

## Features

- Train an AI model based on Binance data for cryptocurrency predictions.
- Select the most accurate model for real-time trading.
- Automate trading for gaining money.

![Login Screen](images/login_screen.png)

![Trading Screen](images/trading_screen.png)

---

## Prerequisites

- Minicoda3: See install [instruction](https://docs.anaconda.com/miniconda/install/#quick-command-line-install).
- ```talib``` for viewing diagrams: See
  install [instruction](https://medium.com/@outwalllife001/how-to-install-ta-lib-on-ubuntu-22-04-step-by-step-88ffd2507bbd).
- Ubuntu 22.04 for optimal performance.
- ```python3``` or ```python3.9```.
- A Binance account with cryptocurrency funds.
- Knowledge about cryptocurrency, trading, python.

---

## Instructions  

### For Real-Time Trading  
1. Clone this repository.  
2. Run the command: ```python3 main.py```.  
3. Log in using your Binance wallet's private and public keys.  
4. Enter the token you want to trade.  
5. Press the "Trade" button and wait for the results.  

### For Training New Models  
1. Clone this repository.  
2. Navigate to the `research` folder and open `gathering_data.ipynb`.  
3. Start the MongoDB service.  
4. Run all cells in `gathering_data.ipynb` to collect sample data.  
5. Run `training_model.ipynb` to train a model. Note the section where the model is saved and save the most compatible model of your choice.  
6. Run `backtest.ipynb` to test your model's performance using sample data.  
7. Run `test_live_trading.ipynb` to test your model's performance with real-time data.  

---  

## Improvements

- Enhanced UI.

---

## Contributors

- Trần Mạnh Duy - 22026567
- Phạm Khánh Linh - 21020080