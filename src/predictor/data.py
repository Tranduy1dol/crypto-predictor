import datetime

import joblib
import numpy as np
import pandas as pd
import ta.momentum as momentum
import ta.trend as trend
import ta.volume as volume
import talib
import tensorflow as tf
from binance.client import Client
from sklearn.preprocessing import MinMaxScaler


def transform(x, y, window_size=30):
    """
    Transforms the input data into sequences of a specified window size for training.

    Args:
        x (ndarray): Input features.
        y (ndarray): Target values.
        window_size (int): Size of the sliding window.

    Returns:
        tuple: Transformed x and y tensors.
    """
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


def simple_process(data):
    """
    Processes the dataset to compute basic technical indicators.

    Args:
        data (DataFrame): Input dataset.

    Returns:
        tuple: Processed dataset and list of selected features.
    """
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

    features = ['close', 'macd', 'rsi', 'mfi', 'so', 'cmf', 'ema']

    return dataset, features


def complex_process(data):
    """
    Processes the dataset to compute a comprehensive set of technical indicators.

    Args:
        data (DataFrame): Input dataset.

    Returns:
        tuple: Processed dataset and list of selected features.
    """
    dataset = data.loc[:, ['time', 'open', 'high', 'low', 'close', 'volume']]

    open = dataset['open']
    high = dataset['high']
    low = dataset['low']
    close = dataset['close']
    volume = dataset['volume']

    avg = (dataset['high'] + dataset['low']) / 2
    dataset['bbands_upperband'], dataset['bbands_middleband'], dataset['bbands_lowerband'] = talib.BBANDS(
        close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
    dataset['bbands_upperband'] = (dataset['bbands_upperband'] - avg) / close
    dataset['bbands_middleband'] = (dataset['bbands_middleband'] - avg) / close
    dataset['bbands_lowerband'] = (dataset['bbands_lowerband'] - avg) / close
    dataset['dema'] = (talib.DEMA(close, timeperiod=30) - avg) / close
    dataset['ema'] = (talib.EMA(close, timeperiod=30) - avg) / close
    dataset['ht_trendline'] = (talib.HT_TRENDLINE(close) - avg) / close
    dataset['kama'] = (talib.KAMA(close, timeperiod=30) - avg) / close
    dataset['ma'] = (talib.MA(close, timeperiod=30, matype=0) - avg) / close
    dataset['midpoint'] = (talib.MIDPOINT(close, timeperiod=14) - avg) / close
    dataset['sma'] = (talib.SMA(close, timeperiod=30) - avg) / close
    dataset['t3'] = (talib.T3(close, timeperiod=5, vfactor=0) - avg) / close
    dataset['tema'] = (talib.TEMA(close, timeperiod=30) - avg) / close
    dataset['trima'] = (talib.TRIMA(close, timeperiod=30) - avg) / close
    dataset['wma'] = (talib.WMA(close, timeperiod=30) - avg) / close
    dataset['linearreg'] = (talib.LINEARREG(close, timeperiod=14) - close) / close
    dataset['linearreg_intercept'] = (talib.LINEARREG_INTERCEPT(
        close, timeperiod=14) - close) / close

    dataset['ad'] = talib.AD(high, low, close, volume) / close
    dataset['adosc'] = talib.ADOSC(high, low, close, volume,
                                   fastperiod=3, slowperiod=10) / close
    dataset['apo'] = talib.APO(close, fastperiod=12,
                               slowperiod=26, matype=0) / close
    dataset['ht_phasor_inphase'], dataset['ht_phasor_quadrature'] = talib.HT_PHASOR(
        close)
    dataset['ht_phasor_inphase'] /= close
    dataset['ht_phasor_quadrature'] /= close
    dataset['linearreg_slope'] = talib.LINEARREG_SLOPE(close, timeperiod=14) / close
    dataset['macd_macd'], dataset['macd_macdsignal'], dataset['macd_macdhist'] = talib.MACD(
        close, fastperiod=12, slowperiod=26, signalperiod=9)
    dataset['macd_macd'] /= close
    dataset['macd_macdsignal'] /= close
    dataset['macd_macdhist'] /= close
    dataset['minus_dm'] = talib.MINUS_DM(high, low, timeperiod=14) / close
    dataset['mom'] = talib.MOM(close, timeperiod=10) / close
    dataset['obv'] = talib.OBV(close, volume) / close
    dataset['plus_dm'] = talib.PLUS_DM(high, low, timeperiod=14) / close
    dataset['stddev'] = talib.STDDEV(close, timeperiod=5, nbdev=1) / close
    dataset['trange'] = talib.TRANGE(high, low, close) / close

    # Momentum Indicators
    dataset['adx'] = talib.ADX(high, low, close, timeperiod=14)
    dataset['adxr'] = talib.ADXR(high, low, close, timeperiod=14)
    dataset['aroon_aroondown'], dataset['aroon_aroonup'] = talib.AROON(
        high, low, timeperiod=14)
    dataset['aroonosc'] = talib.AROONOSC(high, low, timeperiod=14)
    dataset['bop'] = talib.BOP(open, high, low, close)
    dataset['cci'] = talib.CCI(high, low, close, timeperiod=14)
    dataset['dx'] = talib.DX(high, low, close, timeperiod=14)

    dataset['mfi'] = talib.MFI(high, low, close, volume, timeperiod=14)
    dataset['minus_di'] = talib.MINUS_DI(high, low, close, timeperiod=14)
    dataset['plus_di'] = talib.PLUS_DI(high, low, close, timeperiod=14)
    dataset['rsi'] = talib.RSI(close, timeperiod=14)
    dataset['stoch_slowk'], dataset['stoch_slowd'] = talib.STOCH(
        high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    dataset['stochf_fastk'], dataset['stochf_fastd'] = talib.STOCHF(
        high, low, close, fastk_period=5, fastd_period=3, fastd_matype=0)
    dataset['stochrsi_fastk'], dataset['stochrsi_fastd'] = talib.STOCHRSI(
        close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
    dataset['trix'] = talib.TRIX(close, timeperiod=30)
    dataset['ultosc'] = talib.ULTOSC(
        high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
    dataset['willr'] = talib.WILLR(high, low, close, timeperiod=14)

    dataset['atr'] = talib.ATR(high, low, close, timeperiod=14)
    dataset['natr'] = talib.NATR(high, low, close, timeperiod=14)

    dataset['ht_dcperiod'] = talib.HT_DCPERIOD(close)
    dataset['ht_dcphase'] = talib.HT_DCPHASE(close)
    dataset['ht_sine_sine'], dataset['HT_SINE_leadsine'] = talib.HT_SINE(close)
    dataset['ht_trendmode'] = talib.HT_TRENDMODE(close)

    dataset['beta'] = talib.BETA(high, low, timeperiod=5)
    dataset['correl'] = talib.CORREL(high, low, timeperiod=30)

    dataset['linearreg_angle'] = talib.LINEARREG_ANGLE(close, timeperiod=14)

    features = [
        'close',
        'adx',
        'adxr',
        'apo',
        'aroon_aroondown',
        'aroon_aroonup',
        'aroonosc',
        'cci',
        'dx',
        'macd_macd',
        'macd_macdsignal',
        'macd_macdhist',
        'mfi',
        'mom',
        'rsi',
        'stoch_slowk',
        'stoch_slowd',
        'stochf_fastk', 'ultosc', 'willr', 'ht_dcperiod', 'ht_dcphase', 'ht_phasor_inphase', 'ht_phasor_quadrature',
        'ht_trendline', 'beta', 'linearreg', 'linearreg_angle',
        'linearreg_intercept', 'linearreg_slope', 'stddev', 'bbands_upperband', 'bbands_middleband', 'bbands_lowerband',
        'dema', 'ema', 'ht_trendline', 'kama', 'ma', 'midpoint', 't3', 'tema', 'trima', 'wma',
    ]

    dataset.dropna(inplace=True)

    return dataset, features


def normalize(dataset, path, feature):
    """
    Normalizes the dataset features and target values using MinMaxScaler.

    Args:
        dataset (DataFrame): Input dataset.
        path (str): Path to save the scalers.
        feature (list): List of feature column names.

    Returns:
        tuple: Normalized feature and target arrays.
    """
    y = dataset.filter(['close']).values
    x = dataset[feature].values

    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    x = scaler_x.fit_transform(x)
    y = scaler_y.fit_transform(y)
    joblib.dump((scaler_x, scaler_y), path)

    return x, y


class Data:
    """
    Represents the data handling and processing pipeline for a specific symbol.

    Attributes:
        symbol (str): The trading symbol (e.g., BTC).
        start_str (str): Start date for data retrieval.
        end_str (str): End date for data retrieval.
        mode (str): Processing mode ('simple' or 'complex').
    """

    def __init__(self, symbol, start, end, mode):
        """
        Initializes the Data object with symbol, date range, and processing mode.

        Args:
            symbol (str): Trading symbol.
            start (str): Start date.
            end (str): End date.
            mode (str): Processing mode ('simple' or 'complex').
        """
        self.data = None
        self.features = None
        self.symbol = symbol
        self.start_str = start
        self.end_str = end
        self.mode = mode

    def load(self):
        """
        Loads historical data for the specified symbol and date range.
        """
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
        data_frames['time'] = pd.to_datetime(data_frames['time'].astype("int64"), unit="ms")
        data_frames.dropna(inplace=True)

        self.data = data_frames

    def split(self):
        """
        Splits the dataset into training and testing sets.

        Returns:
            tuple: Training and testing datasets.
        """
        data = self.data

        train_size = int(len(data) * 0.8)
        test_data = data.iloc[train_size:]
        train_data = data.iloc[:train_size]
        return train_data, test_data

    def get_features(self):
        """
        Retrieves the processed feature list.

        Returns:
            list: List of feature names.

        Raises:
            ValueError: If features have not been processed yet.
        """
        if self.features is None:
            raise ValueError("Features have not been processed yet.")
        return self.features

    def process(self, data):
        """
        Processes the dataset based on the specified mode.

        Args:
            data (DataFrame): Input dataset.

        Returns:
            DataFrame: Processed dataset.

        Raises:
            ValueError: If an invalid mode is specified.
        """
        if self.mode == 'simple':
            data, features = simple_process(data)
            self.features = features
            return data
        elif self.mode == 'complex':
            data, features = complex_process(data)
            self.features = features
            return data
        else:
            raise ValueError("Invalid mode. Choose 'simple' or 'complex'.")


def make_prediction(model_path, symbol, scaler_path, window_size, mode):
    """
    Makes a prediction for the closing price of a symbol using a trained model.

    Args:
        model_path (str): Path to the trained model.
        symbol (str): Trading symbol.
        scaler_path (str): Path to the saved scalers.
        window_size (int): Size of the input window for the model.
        mode (str): Processing mode ('simple' or 'complex').

    Returns:
        float: Predicted closing price.
    """
    model = tf.keras.models.load_model(model_path)
    model_input_shape = model.input_shape  # (None, timesteps, features)
    window_size = model_input_shape[1]

    today = datetime.date.today().strftime('%b %d %Y')

    dataset = Data(symbol, 'Jan 01 2018', today, mode)
    dataset.load()
    processed_data = dataset.process(dataset.data)

    scaler_x, scaler_y = joblib.load(scaler_path)
    x = processed_data[dataset.get_features()].values
    x = scaler_x.transform(x)

    x_input = x[-window_size:]

    if x_input.shape[0] != window_size:
        raise ValueError(f"Not enough data to make predict. Need {window_size} rows, but {x_input.shape[0]} rows.")

    x_input = np.expand_dims(x_input, axis=0)

    prediction = model.predict(x_input)
    pred_value = prediction.reshape(-1, 1)
    predicted_close = scaler_y.inverse_transform(pred_value)[0, 0]

    return predicted_close
