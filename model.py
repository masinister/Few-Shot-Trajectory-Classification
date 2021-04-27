import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch import autograd
import numpy as np

class Siamese(nn.Module):

    def __init__(self, input_size):
        super(Siamese, self).__init__()
        self.input_size = input_size

        self.rnn = nn.RNN(input_size = self.input_size, hidden_size = 64, num_layers = 2, dropout = 0.1, batch_first = True)

        self.fc = nn.Sequential(nn.Linear(64, 128),
                                nn.Dropout(0.1),
                                nn.ReLU(),
                                nn.Linear(128, 128),
                                nn.Dropout(0.1),
                                nn.ReLU())

        self.out = nn.Sequential(nn.Linear(128, 128),
                                nn.Dropout(0.1),
                                nn.ReLU(),
                                nn.Linear(128, 2))

    def forward_one(self, x):
        x = self.rnn(x, None)[0][:, -1, :]
        x = self.fc(x)
        return x

    def forward(self, x1, x2):
        y1 = self.forward_one(x1)
        y2 = self.forward_one(x2)
        diff = torch.abs(y1 - y2)
        out = self.out(diff)
        return out

    def save(self, path):
        torch.save(self.state_dict(), path)

class SuperSiamese(nn.Module):

    def __init__(self, input_size, split):
        super(SuperSiamese, self).__init__()
        self.input_size = input_size
        self.split = split

        self.rnn1 = nn.RNN(input_size = self.input_size, hidden_size = 256, num_layers = 2, dropout = 0.1, batch_first = True)

        self.rnn2 = nn.RNN(input_size = 256, hidden_size = 256, num_layers = 2, dropout = 0.1, batch_first = True)

        self.fc = nn.Sequential(nn.Linear(256, 512),
                                nn.Dropout(0.1),
                                nn.ReLU(),
                                nn.Linear(512, 512),
                                nn.Dropout(0.1),
                                nn.ReLU())

        self.out = nn.Sequential(nn.Linear(512, 512),
                                nn.Dropout(0.1),
                                nn.ReLU(),
                                nn.Linear(512, 2))

    def forward_one(self, x):
        x = torch.flatten(x, 0, 1)
        x = self.rnn1(x , None)[0][:, -1, :]
        x = torch.stack(torch.split(x, self.split))
        x, _ = torch.sort(x, 1)
        x = self.rnn2(x, None)[0][:, -1, :]
        x = self.fc(x)
        return x

    def forward(self, x1, x2):
        y1 = self.forward_one(x1)
        y2 = self.forward_one(x2)
        diff = torch.abs(y1 - y2)
        out = self.out(diff)
        return out

    def save(self, path):
        torch.save(self.state_dict(), path)

if __name__ == '__main__':
    model = SuperSiamese(17)
    print(model)
