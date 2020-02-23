import MnistIO
import numpy as np


class DataBatch:
    def __init__(self, size, bias, raw_data):
        """
        Uses one hot encoding for the labels of raw_data
        :param size: the size of the batch
        :param bias: the ratio of data labels in the batch
        :param raw_data: data to select the batch from
        """
        self.bias = bias

        # constructing self.data
        sample_num = (size * np.array(bias)).astype(dtype=np.int32)
        sample_num[-1] += size - np.sum(self.samples)

        self.samples = np.array([
            np.random.randint(
                low=0,
                high=len(raw_data[i]),     # number of elements of number i
                size=sample_num[i]
            )
            for i in range(len(raw_data))  # len(raw_data) = 10
        ])

        self.data = np.array([
            raw_data[i][j]
            for i in range(len(raw_data))  # len(raw_data) = 10
            for j in self.samples[i]
        ])

        # constructing self.labels
        self.labels = np.zeros(
            shape=(size, len(raw_data)),
            dtype=np.float32
        )

        index = 0
        for i in range(len(sample_num)):  # len(sample_num) = 10
            for j in range(sample_num[i]):
                self.labels[index+j][i] = 1
            index += sample_num[i]


class DataSelection:
    @staticmethod
    def get_training_data(batch_num: int, size: int, bias: list = None) -> list:
        """
        :param batch_num: number of batches to train on
        :param size: number of images per batch
        :param bias: the proportion of each label to be loaded
        :return: a of tuples of DataBatch objects

        [
            # batch 1
            DataBatch.images = np.array(dtype=np.float32, shape=(size, 28, 28)),
            DataBatch.labels = np.array(dtype=np.float32, shape=(size, 10))

            # batch 2

            ...

            # batch batch_num
            DataBatch.images = np.array(dtype=np.float32, shape=(size, 28, 28)),
            DataBatch.labels = np.array(dtype=np.float32, shape=(size, 10))
        ]
        """
        data, labels = _DataIO.load_training_data()
        sorted_data = [[] * 10]
        for i in range(len(data)):
            sorted_data[labels[i]].append(data[i])

        if bias is None:
            return [
                DataBatch(
                    size=size,
                    bias=[0.1] * 10,
                    raw_data=sorted_data
                )
                for _ in range(batch_num)
            ]

        if (
                type(bias) is list or
                type(bias) is tuple
        ) and (
                len(bias) == 10
        ):
            return [
                DataBatch(
                    size=size,
                    bias=bias,
                    raw_data=sorted_data
                )
                for _ in range(batch_num)
            ]

        raise TypeError("bias argument invalid")

    @staticmethod
    def get_testing_data(size: int, bias: list = None) -> tuple:
        pass


class _DataIO:
    """
    Python wrapper class for C extension
    """

    @staticmethod
    def load_training_data():
        training_data = MnistIO.C_loadTrainingSet()
        if training_data is None:
            raise IOError("Error loading training data")

        training_labels = MnistIO.C_loadTrainingLabels()
        if training_labels is None:
            raise IOError("Error loading training labels")

        return training_data, training_labels

    @staticmethod
    def load_testing_data():
        testing_data = MnistIO.C_loadTrainingSet()
        if testing_data is None:
            raise IOError("Error loading testing data")

        testing_labels = MnistIO.C_loadTestingLabels()
        if testing_labels is None:
            raise IOError("Error loading testing labels")

        return testing_data, testing_labels
