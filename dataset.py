import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from utils import load_data
import numpy as np
import random
from itertools import combinations

class TrajectoryDataset(Dataset):

    def __init__(self, traj_len, num_drivers, device):
        self.device = device
        self.num_drivers = num_drivers
        self.data = load_data(traj_len = traj_len, num_drivers = num_drivers, device = self.device)
        # print(self.data.keys(), [self.data[i].shape for i in self.data.keys()])
        self.length = 50000

    def __getitem__(self, index):
        if index % 2 == 0:
            [i1] = [i2] = random.sample(range(self.num_drivers), 1)
        else:
            [i1, i2] = random.sample(range(self.num_drivers), 2)
        x1 = random.choice(self.data[i2])
        x2 = random.choice(self.data[i1])
        return x1, x2, torch.tensor(int(i1 == i2), dtype=torch.long, device = self.device)

    def __len__(self):
        return self.length

class SuperTrajectoryDataset(Dataset):

    def __init__(self, traj_len, super, num_drivers, device):
        self.device = device
        self.num_drivers = num_drivers
        self.super = super
        self.data = load_data(traj_len = traj_len, num_drivers = num_drivers, device = self.device)
        # print(self.data.keys(), [self.data[i].shape for i in self.data.keys()])
        self.length = 50000

    def __getitem__(self, index):
        if index % 2 == 0:
            [i1] = [i2] = random.sample(range(self.num_drivers), 1)
        else:
            [i1, i2] = random.sample(range(self.num_drivers), 2)
        x1 = random.choices(self.data[i2], k = self.super)
        x2 = random.choices(self.data[i1], k = self.super)
        return torch.stack(x1), torch.stack(x2), torch.tensor(int(i1 == i2), dtype=torch.long, device = self.device)

    def __len__(self):
        return self.length

if __name__ == '__main__':
    print("Dataset test:")
    d = SuperTrajectoryDataset(64, 4, 5, torch.device("cuda"))
    print(len(d))
    x, x1, y = d[0]
    w, w1, z = d[1]
    print(x.shape, x1.shape, y)
    print(w.shape, w1.shape, z)
    p = pack_padded_sequence(x1, [64]*4, batch_first = True)
    print(p)
