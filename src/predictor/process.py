from configparser import ConfigParser
from datetime import datetime

from .data import Data, normalize, transform, make_prediction
from .model import Model, ModelType, setup


class Parser:
    """
    A class to parse and store configuration settings from a config file.
    """

    def __init__(self):
        # Initialize configuration attributes
        self.model_path = None
        self.scaler_path = None
        self.symbol = None
        self.window_size = None
        self.model = None
        self.mode = None
        self.start = None
        self.end = None

    def read_config_file(self, path):
        """
        Reads and parses the configuration file.

        Args:
            path (str): Path to the configuration file.
        """
        config_object = ConfigParser()
        config_object.read(path)
        config = config_object['CONFIG']
        today_str = datetime.today().strftime('%b %d %Y')
        self.model_path = config.get('model_path', './.model/model.keras') or './.model/model.keras'
        self.scaler_path = config.get('scaler_path', './.scaler/scaler.pkl') or './.scaler/scaler.pkl'
        self.symbol = config.get('symbol', 'BTC') or 'BTC'
        self.window_size = int(config.get('window_size', 30) or 30)
        self.model = config.get('model', 'lstm') or 'lstm'
        self.mode = config.get('mode', 'simple') or 'simple'
        self.end = config.get('end', today_str) or today_str
        self.start = config.get('start', 'Jun 01 2018') or 'Jun 01 2018'


def train_and_test(config_path):
    """
    Trains and tests the model based on the configuration file.

    Args:
        config_path (str): Path to the configuration file.
    """
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
    """
    Makes a prediction using the trained model.

    Args:
        config_path (str): Path to the configuration file.
    """
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
