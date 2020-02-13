import MnistIO
import matplotlib.pyplot as plt
import numpy as np


def main():
    print(dir(MnistIO))
    data = MnistIO.C_loadTrainingSet()

    plt.imshow(data[0])
    plt.show()


if __name__ == "__main__":
    main()
