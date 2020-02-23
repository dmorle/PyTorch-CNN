import unittest
import DataIO
from DataIO.DataSelection import _DataIO

import matplotlib.pyplot as plt
import numpy as np


class MyTestCase(unittest.TestCase):
    def test__DataIO(self):
        data, labels = _DataIO.load_training_data()

        self.assertEqual(len(data), 60000, "images did not load correctly")
        self.assertEqual(len(labels), 60000, "Labels did not load correctly")

        plt.imshow(data[0])
        plt.show()

    def test_DataBatch(self):
        DataIO.DataSelection.get_training_data(1, 100)

        self.assertEqual(True, False)

        return


if __name__ == '__main__':
    unittest.main()
