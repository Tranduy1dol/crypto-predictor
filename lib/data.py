import datetime

import joblib
import numpy as np
import pandas as pd
import ta.momentum as momentum
import ta.trend as trend
import ta.volume as volume
import tensorflow as tf
from binance.client import Client
from sklearn.preprocessing import MinMaxScaler


def transform(x, y, window_size):
    x_train = []
    y_train = []

    for i in range(window_size, len(x) - 2):
        x_train.append(x[i - window_size:i, :])
        y_train.append(y[i + 2, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
    y_train = np.reshape(y_train, (-1, 1))

    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)

    return x_train, y_train


def process(data):
    dataset = data.loc[:, ['time', 'open', 'high', 'low', 'close', 'volume']]

    dataset['rsi'] = momentum.RSIIndicator(dataset['close']).rsi()
    dataset['adx'] = trend.adx(high=dataset['high'], low=dataset['low'], close=dataset['close'])
    dataset['ema'] = trend.ema_indicator(close=dataset['close'])
    dataset['macd'] = trend.macd_diff(close=dataset['close'])
    dataset['so'] = momentum.stoch(high=dataset['high'], low=dataset['low'], close=dataset['close'])
    dataset['vwap'] = volume.volume_weighted_average_price(
        high=dataset['high'],
        low=dataset['low'],
        close=dataset['close'],
        volume=dataset['volume']
    )
    dataset['mfi'] = volume.money_flow_index(
        high=dataset['high'],
        low=dataset['low'],
        close=dataset['close'],
        volume=dataset['volume']
    )
    dataset['cmf'] = volume.chaikin_money_flow(
        high=dataset['high'],
        low=dataset['low'],
        close=dataset['close'],
        volume=dataset['volume']
    )

    dataset.dropna(inplace=True)

    return dataset


def normalize(dataset, path):
    y = dataset.filter(['close']).values
    x = dataset[['close', 'macd', 'rsi', 'mfi', 'so', 'cmf', 'ema']].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    x = scaler.fit_transform(x)
    y = scaler.fit_transform(y)
    joblib.dump(scaler, path)

    return x, y


class Data:
    def __init__(self, symbol, start_str, end_str):
        self.data = None
        self.symbol = symbol
        self.start_str = start_str
        self.end_str = end_str

    def load(self):
        client = Client("", "")
        klines = client.get_historical_klines(
            symbol=f'{self.symbol}USDT',
            interval=Client.KLINE_INTERVAL_1DAY,
            start_str=self.start_str,
            end_str=self.end_str
        )

        data_frames = pd.DataFrame(klines)
        data_frames = data_frames.iloc[:, :6]
        data_frames.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
        data_frames = data_frames.astype(float)
        data_frames['time'] = pd.to_datetime(data_frames['time'])
        data_frames.dropna()

        self.data = data_frames

    def split(self):
        data = self.data

        train_size = int(len(data) * 0.8)
        test_data = data.iloc[train_size:]
        train_data = data.iloc[:train_size]
        return train_data, test_data


def make_prediction(model_path, symbol, scaler_path, window_size):
    """
    Predict the close price for tomorrow using the trained model.

    :param model_path: Path to the trained model file.
    :param symbol: Cryptocurrency symbol (e.g., 'BTC').
    :param scaler_path: Path to the scaler file.
    :param window_size: Number of days for the sliding window.
    :return: Predicted close price for tomorrow.
    """
    today = datetime.datetime.utcnow().strftime('%b %d %Y')
    dataset = Data(symbol, 'Jan 01 2018', today)
    dataset.load()
    processed_data = process(dataset.data)

    scaler = joblib.load(scaler_path)
    x = processed_data[['close', 'macd', 'rsi', 'mfi', 'so', 'cmf', 'ema']].values
    x = scaler.fit_transform(x)

    x_input = x[-window_size:]
    x_input = np.expand_dims(x_input, axis=0)

    model = tf.keras.models.load_model(model_path)
    prediction = model.predict(x_input)

    predicted_close = scaler.inverse_transform(
        np.concatenate((prediction, np.zeros((prediction.shape[0], x.shape[1] - 1))), axis=1)
    )[0, 0]

    return predicted_close
