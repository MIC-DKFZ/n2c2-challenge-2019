import torch

from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, ReLU, Linear, Sequential
import numpy as np


def test_fun(x):
    return 2*x+3 + 0.1 * (2*np.random.rand()-1)/2


class TestNet(nn.Module):

    def __init__(self):

        super().__init__(D_in, H, D_out)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(D_in, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, D_out),
        )


    def forward(self, hidden_states):
        out = self.model(hidden_states)
        out = out.view(out.size(0), -1)
        return out


if __name__ == '__main__':
    x = np.arange(0, 10, 0.2)
    y = test_fun(x)

    test_net = TestNet()

    batch_size = 128

    input_data = torch.tensor(np.arange(0, 10, 1), dtype=torch.float32).repeat(batch_size)
    input_data = input_data.view(-1, 10)

    print(input_data)
    input_data = test_net(input_data)
    print(input_data)