import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import os

from model import Siamese, SuperSiamese
from dataset import TrajectoryDataset, SuperTrajectoryDataset
from learning import train, test
from utils import train_test_split

device = torch.device("cuda")
num_drivers = 500
traj_len = 64
super = 8
n_features = 17
test_split = 0.2
batch_size = 500
num_epochs = 500

# dataset = TrajectoryDataset(traj_len = traj_len, num_drivers = num_drivers, device = device)
# model = Siamese(input_size = n_features).to(device)

dataset = SuperTrajectoryDataset(traj_len = traj_len, super = super, num_drivers = num_drivers, device = device)
model = SuperSiamese(input_size = n_features, split = super).to(device)

trainloader, testloader = train_test_split(dataset, test_split, batch_size)

model = train(model, trainloader, testloader, device, num_epochs)

model.save('model.pth')
