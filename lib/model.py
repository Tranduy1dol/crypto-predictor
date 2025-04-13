from enum import Enum

import tensorflow as tf
from keras.src.layers import Bidirectional, GRU
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.models import Sequential


class ModelType(Enum):
    LSTM = 1
    GRU = 2
    BI_LSTM = 3


def setup():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)


def save_model(model, path):
    model.save(path)


class Model:
    def __init__(self, model_type: ModelType):
        self.model_type = model_type

    def train(self, x_train, y_train):
        model = Sequential()

        match self.model_type.value:
            case 1:
                model.add(LSTM(50, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2])))
            case 2:
                model.add(Bidirectional(LSTM(50, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2]))))
            case 3:
                model.add(GRU(50, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2])))

        model.compile(loss='mean_squared_error', optimizer='adam')
        history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)
        return model, history

    def test(self):
        # Placeholder for testing logic
        pass
