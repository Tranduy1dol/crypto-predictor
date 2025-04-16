from configparser import ConfigParser

from .data import Data, normalize, transform, make_prediction
from .model import Model, ModelType, setup


class Parser:
    def __init__(self):
        self.model_path = None
        self.scaler_path = None
        self.symbol = None
        self.window_size = None
        self.model = None
        self.mode = None
        self.start = None
        self.end = None

    def read_config_file(self, path):
        config_object = ConfigParser()
        config_object.read(path)

        config = config_object['CONFIG']

        self.model_path = config['model_path']
        self.scaler_path = config['scaler_path']
        self.symbol = config['symbol']
        self.window_size = int(config['window_size'])
        self.model = config['model']
        self.mode = config['mode']
        self.end = config['end']
        self.start = config['start']
        self.symbol = config['symbol']


def train_and_test(config_path):
    # Parse the configuration file
    parser = Parser()
    parser.read_config_file(config_path)

    # Load and process data
    data = Data(parser.symbol, parser.start, parser.end, parser.mode)
    data.load()
    train_data, test_data = data.split()
    train_processed = data.process(train_data)
    test_processed = data.process(test_data)
    features = data.get_features()

    # Normalize and transform data
    x_train, y_train = normalize(train_processed, parser.scaler_path, features)
    x_train, y_train = transform(x_train, y_train)

    # Setup GPU and train the model
    print("Training")
    setup()
    model_type = ModelType(parser.model.lower())
    model = Model(model_type)
    model.train(x_train, y_train, parser.model_path)

    # Test the model
    model.test(test_processed, parser.scaler_path, features)
    model.result()


def predict(config_path):
    # Parse the configuration file
    parser = Parser()
    parser.read_config_file(config_path)

    # Make a prediction
    predicted_close = make_prediction(
        parser.model_path,
        parser.symbol,
        parser.scaler_path,
        parser.window_size,
        parser.mode
    )
    print(f"Predicted Close Price: {predicted_close}")
