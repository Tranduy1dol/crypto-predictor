import unittest

from lib.data import Data, transform


class TestData(unittest.TestCase):
    def test(self):
        dataset = Data('BTC', 'Jan 01 2018', 'Jun 30 2021')

        dataset.load()
        print(dataset.data)

        dataset.describe()
        dataset.data_info()

        x, y = dataset.normalize()
        x_train, y_train = transform(x, y)

        print(x_train.shape)
        print(y_train.shape)

if __name__ == '__main__':
    unittest.main()
