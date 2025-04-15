import unittest
from lib import make_prediction


class TestData(unittest.TestCase):
    def test_make_prediction(self):
        scaler_path = './.scaler/scaler.pkl'
        model_path = './.model/model.keras'
        symbol = "BTC"
        window_size = 10

        try:
            prediction = make_prediction(model_path, symbol, scaler_path, window_size)
            self.assertIsInstance(prediction, float, "Prediction should be a float value.")
            self.assertGreater(prediction, 0, "Prediction should be a positive value.")
        except Exception as e:
            self.fail(f"make_prediction raised an exception: {e}")

        print(prediction)


if __name__ == '__main__':
    unittest.main()
