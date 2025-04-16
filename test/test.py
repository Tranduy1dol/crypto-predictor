import unittest

from src.predictor.data import Data, normalize, transform
from src.predictor.model import setup, ModelType, Model


class TestTrainAndTest(unittest.TestCase):
    def test_model(self):
        scaler_path = 'scaler.pkl'
        model_path = 'model.keras'

        data = Data('BTC', 'Jun 01 2018', 'Jun 01 2023', 'simple')
        data.load()
        train_data, test_data = data.split()
        train_processed = data.process(train_data)
        test_processed = data.process(test_data)
        features = data.get_features()

        # Normalize and transform data
        x_train, y_train = normalize(train_processed, scaler_path, features)
        x_train, y_train = transform(x_train, y_train)

        # Setup GPU and train the model
        setup()
        model_type = ModelType('lstm')
        model = Model(model_type)
        model.train(x_train, y_train, model_path)

        # Test the model
        model.test(test_processed, scaler_path, features)
        model.result()

if __name__ == '__main__':
    unittest.main()
