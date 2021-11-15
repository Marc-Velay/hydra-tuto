import torch
from torchvision import datasets
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

from utils.abstracts.dataset import Dataset

def integer_sqrt(n):
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x

class MNIST(Dataset):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args

        train_loader = self.get_loader(train=True, batch_size=int(self.args["train_batch_size"]))
        test_loader = self.get_loader(train=False, batch_size=int(self.args["test_batch_size"]))

        batch_idx, (example_data, example_targets) = next(enumerate(test_loader))

        example_data = example_data.cpu()

        self.render(data=example_data, y_true=example_targets, shape=(3, 4))

    def get_loader(self, train=True, batch_size=16, shuffle=True):
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=train, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,)) # Mean and Std of the MNIST dataset
                       ])),
        batch_size=batch_size, shuffle=shuffle)

        return train_loader

    def reset(self):
        return super().reset()

    def render(self, data, y_true, y_pred=None, shape=(2, 3)):
        rows = integer_sqrt(len(data))
        cols = len(data)//rows
        remains = len(data) % rows
        if np.prod((rows, cols)) < np.prod(shape):
            shape  = (rows+1, cols)
            nb_plot = np.prod((rows, cols))+remains
        else:
            nb_plot = np.prod(shape)

        for i in range(nb_plot):
            plt.subplot(*shape, i+1)
            plt.tight_layout()
            print("---------------------------------")
            print(i, data.shape)
            plt.imshow(data[i][0], cmap='gray', interpolation='none')
            plt.title("True: {}".format(y_true[i]))
            plt.xticks([])
            plt.yticks([])
        plt.show()

