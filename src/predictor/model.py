from enum import Enum

import joblib
import numpy as np
import tensorflow as tf
from keras.src.layers import LSTM, Bidirectional, GRU, Dense
from keras.src.models import Sequential
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


class ModelType(Enum):
    """
    Enum for different types of models.
    """
    LSTM = 'lstm'
    GRU = 'gru'
    BI_LSTM = 'bi_lstm'


def setup():
    """
    Configures TensorFlow to use GPU with memory growth.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)


class Model:
    """
    A class to define, train, test, and evaluate machine learning models.
    """

    def __init__(self, model_type: ModelType):
        # Initialize model attributes
        self.model_type = model_type
        self.model = None
        self.y_pred = None
        self.y_test_label = None

    def train(self, x_train, y_train, path):
        """
        Trains the model using the provided training data.

        Args:
            x_train (ndarray): Training features.
            y_train (ndarray): Training targets.
            path (str): Path to save the trained model.
        """
        with tf.device('/GPU:0'):
            model = Sequential()

        match self.model_type.value:
            case 'lstm':
                model.add(LSTM(256, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
                model.add(LSTM(64, return_sequences=False))
            case 'bi_lstm':
                model.add(
                    Bidirectional(LSTM(256, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2]))))
                model.add(Bidirectional(LSTM(64, return_sequences=False)))
            case 'gru':
                model.add(GRU(256, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
                model.add(GRU(64, return_sequences=False))

        model.add(Dense(32))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(x_train, y_train, batch_size=64, epochs=50)

        self.model = model
        model.save(path)

    def test(self, test_data, scaler_path, feature, window_size=30):
        """
        Tests the model on the test dataset.

        Args:
            test_data (DataFrame): Test dataset.
            scaler_path (str): Path to the scaler file.
            feature (list): List of feature columns.
            window_size (int): Size of the sliding window.
        """
        x_test = test_data[feature].values
        y_test = test_data.filter(['close']).values

        idx = np.arange(len(test_data))

        x_test_zero_time = x_test[idx]

        scaler_x, scaler_y = joblib.load(scaler_path)
        x_test = scaler_x.transform(x_test_zero_time)

        x_test_sliding = []
        y_test_label = []

        for i in range(window_size, len(x_test)):
            x_test_sliding.append(x_test[i - window_size:i])
        for i in range(window_size, len(x_test) - 2):
            y_test_label.append(y_test[i + 2, 0])

        x_test = np.array(x_test_sliding)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

        y_pred = self.model.predict(x_test)
        y_pred = scaler_y.inverse_transform(y_pred)

        self.y_pred, self.y_test_label = y_pred, y_test_label

    def result(self, title="Model Test Result"):
        """
        Displays the test results and evaluation metrics.

        Args:
            title (str): Title for the result plot.

        Returns:
            tuple: MAPE and RMSE metrics.
        """
        y_pred = np.array(self.y_pred).flatten()
        y_true = np.array(self.y_test_label).flatten()

        min_len = min(len(y_pred), len(y_true))
        y_pred = y_pred[:min_len]
        y_test_label = y_true[:min_len]

        mape = mean_absolute_percentage_error(y_test_label, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_label, y_pred))

        plt.figure(figsize=(14, 6))
        plt.plot(y_true, label="Actual", color='blue')
        plt.plot(y_pred, label="Predicted", color='orange')
        plt.title(title)
        plt.xlabel("Time Step")
        plt.ylabel("Close Price")
        plt.legend()
        plt.grid(True)

        plt.figtext(0.5, 0, f"MAPE: {mape:.4f} | RMSE: {rmse:.4f}", ha="center", fontsize=10,
                    bbox={"facecolor": "white", "alpha": 0.5, "pad": 5})

        plt.tight_layout()
        plt.show()
        return mape, rmse
